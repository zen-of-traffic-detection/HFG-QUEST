from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch

from GAT1 import GAT as Stage1GAT
from v2_common import (
    collect_graph_records,
    dump_json,
    evaluate_graph_classifier,
    index_subset,
    oversample_indices,
    safe_stratified_split,
    summarize_token_lengths,
    train_graph_classifier,
    write_report,
)
from v2_dataset import RAW_CLASSES, STAGE1_CLASSES, raw_to_stage1_id
from v2_robustness import (
    build_stage1_perturbed_graphs,
    clean_sample_meta,
    class_pcap_entries,
    evaluate_with_predictions,
    metric_deltas,
    robustness_enabled,
    robustness_ratio,
    robustness_seed,
    robustness_variants,
    validate_pcap_entries,
    write_variant_outputs,
)


def stage1_n_packets_from_config(config: Dict[str, object]) -> int:
    preprocess_cfg = config["preprocess"]  # type: ignore[index]
    return int(preprocess_cfg.get("stage1_n_packets", preprocess_cfg["n_packets"]))


def stage1_packet_offset_from_config(config: Dict[str, object]) -> int:
    preprocess_cfg = config["preprocess"]  # type: ignore[index]
    return max(0, int(preprocess_cfg.get("stage1_packet_offset", 0)))


def run_stage1(config: Dict[str, object], before_dir: Path, run_dir: Path, logger) -> Tuple[Dict[str, object], Path]:
    txt_map = {class_name: before_dir / f"{class_name}.txt" for class_name in RAW_CLASSES}
    n_packets = stage1_n_packets_from_config(config)
    packet_offset = stage1_packet_offset_from_config(config)
    records, raw_counts = collect_graph_records(
        txt_map,
        RAW_CLASSES,
        raw_to_stage1_id,
        n_packets,
        packet_offset=packet_offset,
    )
    graphs = [record.graph for record in records]
    labels = [record.label for record in records]
    if not graphs:
        raise RuntimeError("Stage I found no graphs.")
    raw_lengths = [record.raw_token_count for record in records]
    sliced_lengths = [record.sliced_token_count for record in records]
    token_length_summary = {
        "raw_tokens": summarize_token_lengths(raw_lengths),
        "sliced_tokens": summarize_token_lengths(sliced_lengths),
    }
    if n_packets > 0:
        token_length_summary["sliced_tokens"]["full_window_rate"] = float(
            sum(1 for value in sliced_lengths if value >= n_packets) / len(sliced_lengths)
        ) if sliced_lengths else 0.0
    else:
        token_length_summary["sliced_tokens"]["full_window_rate"] = None
    logger.info(
        "Stage I graph window offset=%d n_packets=%d records=%d sliced_mean=%.2f",
        packet_offset,
        n_packets,
        len(graphs),
        float(token_length_summary["sliced_tokens"].get("mean", 0.0)),
    )

    training_cfg = config["training"]  # type: ignore[index]
    split_cfg = training_cfg["split"]
    seed = int(training_cfg["seed"])
    train_idx, test_idx = safe_stratified_split(labels, float(split_cfg["test_size"]), seed)
    val_size = float(split_cfg.get("val_size", 0.0))
    val_idx = []

    train_idx_full = train_idx
    if val_size > 0.0 and train_idx_full:
        inner_train, inner_val = safe_stratified_split([labels[idx] for idx in train_idx_full], val_size, seed + 1)
        val_idx = [train_idx_full[idx] for idx in inner_val]
        train_idx = [train_idx_full[idx] for idx in inner_train]

    oversampling_cfg = training_cfg["oversampling"]
    if bool(oversampling_cfg["enabled"]):
        train_subset_labels = [labels[idx] for idx in train_idx]
        sampled = oversample_indices(train_subset_labels)
        train_graphs = [graphs[train_idx[idx]] for idx in sampled]
    else:
        train_graphs = index_subset(graphs, train_idx)

    val_graphs = index_subset(graphs, val_idx)
    test_graphs = index_subset(graphs, test_idx)

    stage_cfg = training_cfg["stage1"]
    model_cfg = config["model"]["stage1"]  # type: ignore[index]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Stage1GAT(
        in_channels=2,
        hidden_channels=int(model_cfg["hidden_channels"]),
        num_classes=len(STAGE1_CLASSES),
        num_heads=int(model_cfg["num_heads"]),
        dropout_rate=float(model_cfg["dropout_rate"]),
    )

    output_dir = run_dir / "stage1"
    trained = train_graph_classifier(
        model=model,
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        batch_size=int(stage_cfg["batch_size"]),
        epochs=int(stage_cfg["epochs"]),
        lr=float(stage_cfg["lr"]),
        device=device,
        logger=logger,
        ckpt_path=output_dir / "model_best.pth",
    )
    metrics = evaluate_graph_classifier(
        trained,
        test_graphs=test_graphs,
        class_names=STAGE1_CLASSES,
        batch_size=int(stage_cfg["batch_size"]),
        device=device,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(output_dir / "metrics.json", metrics)

    robustness_summary = None
    if robustness_enabled(config):
        entries = class_pcap_entries(config, RAW_CLASSES)
        validate_pcap_entries(records, entries)
        clean_meta = clean_sample_meta(entries, test_idx)
        metrics_by_variant = {}
        variant_summaries = {}
        clean_metrics, clean_true, clean_pred = evaluate_with_predictions(
            trained,
            test_graphs,
            STAGE1_CLASSES,
            int(stage_cfg["batch_size"]),
            device,
        )
        write_variant_outputs(output_dir, "clean_test", STAGE1_CLASSES, clean_metrics, clean_true, clean_pred, clean_meta)
        metrics_by_variant["clean_test"] = clean_metrics
        variant_summaries["clean_test"] = {
            "variant": "clean_test",
            "ratio": 0.0,
            "samples": len(test_idx),
            "changed_samples": 0,
            "unchanged_samples": len(test_idx),
            "notes": ["clean baseline"],
            "final_sliced_token_count_stats": token_length_summary["sliced_tokens"],
        }
        for variant in robustness_variants(config):
            if variant == "clean_test":
                continue
            perturbed_graphs, sample_meta, variant_summary = build_stage1_perturbed_graphs(
                config,
                records,
                test_idx,
                variant,
                RAW_CLASSES,
            )
            variant_metrics, y_true, y_pred = evaluate_with_predictions(
                trained,
                perturbed_graphs,
                STAGE1_CLASSES,
                int(stage_cfg["batch_size"]),
                device,
            )
            write_variant_outputs(output_dir, variant, STAGE1_CLASSES, variant_metrics, y_true, y_pred, sample_meta)
            metrics_by_variant[variant] = variant_metrics
            variant_summaries[variant] = variant_summary
            logger.info(
                "Stage I robustness %s acc=%.4f macro_f1=%.4f min_f1=%.4f changed=%d/%d",
                variant,
                variant_metrics["accuracy"],
                variant_metrics["macro_f1"],
                variant_metrics.get("min_class_f1", 0.0),
                variant_summary["changed_samples"],
                variant_summary["samples"],
            )
        robustness_summary = {
            "enabled": True,
            "applies_to": "test_only",
            "train_perturbed": False,
            "ratio": robustness_ratio(config),
            "seed": robustness_seed(config),
            "variants": variant_summaries,
            "deltas_vs_clean_test": metric_deltas(metrics_by_variant),
        }
        dump_json(output_dir / "perturbation_summary.json", robustness_summary)

    split_summary = {
        "raw_counts": raw_counts,
        "train_graphs": len(train_graphs),
        "val_graphs": len(val_graphs),
        "test_graphs": len(test_graphs),
        "oversampling_enabled": bool(oversampling_cfg["enabled"]),
        "n_packets": n_packets,
        "stage1_n_packets": n_packets,
        "stage1_packet_offset": packet_offset,
        "token_length_summary": token_length_summary,
    }
    return metrics, output_dir
