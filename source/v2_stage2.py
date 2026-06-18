from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch_geometric.loader import DataLoader

from GAT2 import GAT as Stage2GAT
from v2_common import (
    GraphRecord,
    SEQUENCE_GLOBAL_FEATURE_DIM,
    augment_training_records,
    collect_graph_records,
    collect_graphs,
    compute_class_weights,
    dump_json,
    evaluate_graph_classifier,
    graph_record_token_summary,
    graph_label_count_dict,
    index_subset,
    label_count_dict,
    oversample_indices,
    safe_stratified_split,
    set_seed,
    train_graph_classifier,
    write_report,
)
from v2_dataset import MALICIOUS_CLASSES, STAGE2_CLASSES, STAGE2_TO_ID, raw_to_stage2_id
from v2_robustness import (
    build_stage2_perturbed_graphs,
    clean_sample_meta,
    class_pcap_entries,
    metric_deltas,
    robustness_enabled,
    robustness_ratio,
    robustness_seed,
    robustness_variants,
    validate_pcap_entries,
    write_variant_outputs,
)


STAGE2_TOOLS: Tuple[str, ...] = ("deimos", "merlin", "sliver")
STAGE2_HEAD_SEED_OFFSETS = {"deimos": 101, "merlin": 202, "sliver": 303}


def stage2_n_packets_from_config(config: Dict[str, object]) -> int:
    preprocess_cfg = config["preprocess"]  # type: ignore[index]
    return int(preprocess_cfg.get("stage2_n_packets", preprocess_cfg["n_packets"]))


def stage2_packet_offset_from_config(config: Dict[str, object]) -> int:
    preprocess_cfg = config["preprocess"]  # type: ignore[index]
    return max(0, int(preprocess_cfg.get("stage2_packet_offset", 0)))


def stage2_classifier_from_config(config: Dict[str, object]) -> str:
    stage2_cfg = config.get("stage2", {})
    if isinstance(stage2_cfg, dict):
        return str(stage2_cfg.get("classifier", "single_head"))
    return "single_head"


def sequence_global_feature_config(config: Dict[str, object]) -> Tuple[bool, float, int]:
    features_cfg = config.get("features", {})
    sequence_cfg = {}
    if isinstance(features_cfg, dict):
        sequence_cfg = features_cfg.get("sequence_global", {})  # type: ignore[assignment]
    if not isinstance(sequence_cfg, dict):
        sequence_cfg = {}
    enabled = bool(sequence_cfg.get("enabled", False))
    size_clip = float(sequence_cfg.get("size_clip", 1500.0))
    dim = SEQUENCE_GLOBAL_FEATURE_DIM if enabled else 0
    return enabled, size_clip, dim


def graph_node_feature_config(config: Dict[str, object]) -> Tuple[str, float, bool]:
    features_cfg = config.get("features", {})
    node_cfg = {}
    if isinstance(features_cfg, dict):
        node_cfg = features_cfg.get("graph_node", {})  # type: ignore[assignment]
    if not isinstance(node_cfg, dict):
        node_cfg = {}
    transform = str(node_cfg.get("size_transform", "raw"))
    size_clip = float(node_cfg.get("size_clip", 1500.0))
    normalize_index = bool(node_cfg.get("normalize_index", False))
    return transform, size_clip, normalize_index


def tool_classes(tool: str) -> List[str]:
    prefix = f"{tool}_"
    return [class_name for class_name in MALICIOUS_CLASSES if class_name.startswith(prefix)]


def local_label_resolver(class_names: Sequence[str]):
    mapping = {class_name: idx for idx, class_name in enumerate(class_names)}
    return lambda class_name: mapping[class_name]


def metrics_from_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> Dict[str, object]:
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "min_class_f1": float(np.min(f1)) if len(f1) else 0.0,
        "class_metrics": {
            class_names[idx]: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
            for idx in labels
        },
        "confusion_matrix": cm.tolist(),
    }


def evaluate_graph_classifier_with_predictions(
    model: torch.nn.Module,
    test_graphs: Sequence[object],
    class_names: Sequence[str],
    batch_size: int,
    device: torch.device,
) -> Tuple[Dict[str, object], List[int], List[int]]:
    if not test_graphs:
        raise RuntimeError("No test graphs available for evaluation.")
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    loader = DataLoader(list(test_graphs), batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"))
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(pred)
            y_true.extend(batch.y.cpu().numpy().tolist())
    return metrics_from_predictions(y_true, y_pred, class_names), y_true, y_pred


def make_splits(
    records: Sequence[GraphRecord],
    labels: Sequence[int],
    class_names: Sequence[str],
    training_cfg: Dict[str, object],
    n_packets: int,
    include_sequence_features: bool,
    sequence_feature_size_clip: float,
    node_size_transform: str,
    node_size_clip: float,
    normalize_node_index: bool,
    packet_offset: int,
) -> Tuple[List[object], List[object], List[object], List[int], List[int], List[int], Dict[str, object]]:
    split_cfg = training_cfg["split"]  # type: ignore[index]
    seed = int(training_cfg["seed"])
    train_idx, test_idx = safe_stratified_split(labels, float(split_cfg["test_size"]), seed)
    val_size = float(split_cfg.get("val_size", 0.0))
    val_idx: List[int] = []
    if val_size > 0.0 and train_idx:
        inner_train, inner_val = safe_stratified_split([labels[idx] for idx in train_idx], val_size, seed + 1)
        val_idx = [train_idx[idx] for idx in inner_val]
        train_idx = [train_idx[idx] for idx in inner_train]

    graphs = [record.graph for record in records]
    base_train_graphs = index_subset(graphs, train_idx)
    base_train_labels = [labels[idx] for idx in train_idx]
    stage_cfg = training_cfg["stage2"]  # type: ignore[index]
    augmentation_cfg = stage_cfg.get("augmentation", {})
    if not isinstance(augmentation_cfg, dict):
        augmentation_cfg = {}
    if bool(augmentation_cfg.get("enabled", False)):
        augmented_graphs, augmented_labels, augmentation_summary = augment_training_records(
            records=records,
            train_indices=train_idx,
            class_names=class_names,
            n_packets=n_packets,
            include_sequence_features=include_sequence_features,
            sequence_feature_size_clip=sequence_feature_size_clip,
            node_size_transform=node_size_transform,
            node_size_clip=node_size_clip,
            normalize_node_index=normalize_node_index,
            packet_offset=packet_offset,
            augmentation_cfg=augmentation_cfg,
            seed=seed,
        )
    else:
        augmented_graphs = []
        augmented_labels = []
        augmentation_summary = {
            "enabled": False,
            "targets": {},
            "variants_per_sample": 0,
            "augmented_graphs": 0,
            "augmented_by_class": {class_name: 0 for class_name in class_names},
            "skipped_by_class": {class_name: 0 for class_name in class_names},
        }

    train_pool_graphs = base_train_graphs + augmented_graphs
    train_pool_labels = base_train_labels + augmented_labels
    oversampling_cfg = training_cfg["oversampling"]  # type: ignore[index]
    if bool(oversampling_cfg["enabled"]):
        sampled = oversample_indices(train_pool_labels)
        train_graphs = [train_pool_graphs[idx] for idx in sampled]
    else:
        train_graphs = train_pool_graphs

    val_graphs = index_subset(graphs, val_idx)
    test_graphs = index_subset(graphs, test_idx)
    split_summary = {
        "base_train_graphs": len(base_train_graphs),
        "train_pool_graphs_before_oversampling": len(train_pool_graphs),
        "train_graphs": len(train_graphs),
        "val_graphs": len(val_graphs),
        "test_graphs": len(test_graphs),
        "base_train_counts": label_count_dict(base_train_labels, class_names),
        "train_pool_counts_before_oversampling": label_count_dict(train_pool_labels, class_names),
        "final_train_counts": graph_label_count_dict(train_graphs, class_names),
        "val_counts": graph_label_count_dict(val_graphs, class_names),
        "test_counts": graph_label_count_dict(test_graphs, class_names),
        "oversampling_enabled": bool(oversampling_cfg["enabled"]),
        "stage2_packet_offset": packet_offset,
        "augmentation": augmentation_summary,
    }
    return train_graphs, val_graphs, test_graphs, base_train_labels, train_pool_labels, test_idx, split_summary


def stage2_training_options(
    stage_cfg: Dict[str, object],
    train_pool_labels: Sequence[int],
    num_classes: int,
    base_train_labels: Optional[Sequence[int]] = None,
) -> Tuple[str, Optional[torch.Tensor], Dict[str, object]]:
    loss_name = str(stage_cfg.get("loss", "nll"))
    checkpoint_metric = str(stage_cfg.get("checkpoint_metric", "val_loss"))
    checkpoint_macro_weight = float(stage_cfg.get("checkpoint_macro_weight", 0.7))
    focal_gamma = float(stage_cfg.get("focal_gamma", 2.0))
    class_weight_power = float(stage_cfg.get("class_weight_power", 0.5))
    class_weight_source = str(stage_cfg.get("class_weight_source", "train_pool"))
    if class_weight_source == "base_train" and base_train_labels is not None:
        weight_labels = base_train_labels
    else:
        weight_labels = train_pool_labels
    if loss_name.lower() in ("weighted_nll", "focal", "weighted_focal"):
        class_weights, weight_summary = compute_class_weights(weight_labels, num_classes, class_weight_power)
    else:
        class_weights = None
        weight_summary = {
            "power": class_weight_power,
            "counts": [0 for _ in range(num_classes)],
            "weights": [1.0 for _ in range(num_classes)],
        }
    summary = {
        "loss": loss_name,
        "checkpoint_metric": checkpoint_metric,
        "checkpoint_macro_weight": checkpoint_macro_weight,
        "focal_gamma": focal_gamma,
        "class_weight_power": class_weight_power,
        "class_weight_source": class_weight_source,
        "class_weights": weight_summary,
    }
    return loss_name, class_weights, summary


def run_stage2_tool_heads(
    config: Dict[str, object],
    after_dir: Path,
    run_dir: Path,
    logger,
) -> Tuple[Dict[str, object], Path]:
    n_packets = stage2_n_packets_from_config(config)
    packet_offset = stage2_packet_offset_from_config(config)
    training_cfg = config["training"]  # type: ignore[index]
    stage_cfg = training_cfg["stage2"]  # type: ignore[index]
    model_cfg = config["model"]["stage2"]  # type: ignore[index]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = run_dir / "stage2"
    include_sequence_features, sequence_feature_size_clip, global_feature_dim = sequence_global_feature_config(config)
    node_size_transform, node_size_clip, normalize_node_index = graph_node_feature_config(config)

    aggregate_true: List[int] = []
    aggregate_pred: List[int] = []
    per_head_split: Dict[str, object] = {}
    head_metrics: Dict[str, object] = {}
    base_seed = int(training_cfg["seed"])
    robustness_on = robustness_enabled(config)
    robustness_variant_names = robustness_variants(config) if robustness_on else []
    if robustness_on and "clean_test" not in robustness_variant_names:
        robustness_variant_names = ["clean_test", *robustness_variant_names]
    variant_aggregate_true: Dict[str, List[int]] = {variant: [] for variant in robustness_variant_names}
    variant_aggregate_pred: Dict[str, List[int]] = {variant: [] for variant in robustness_variant_names}
    variant_aggregate_meta: Dict[str, List[Dict[str, object]]] = {variant: [] for variant in robustness_variant_names}
    variant_summaries: Dict[str, Dict[str, object]] = {variant: {} for variant in robustness_variant_names}

    for tool in STAGE2_TOOLS:
        head_seed_offset = int(STAGE2_HEAD_SEED_OFFSETS.get(tool, 0))
        head_seed = base_seed + head_seed_offset
        set_seed(head_seed)
        logger.info("Stage II %s head seed reset to %d", tool, head_seed)
        class_names = tool_classes(tool)
        if not class_names:
            raise RuntimeError(f"Stage II tool head found no classes for {tool}.")
        txt_map = {class_name: after_dir / f"{class_name}_packets.txt" for class_name in class_names}
        records, raw_counts = collect_graph_records(
            txt_map,
            class_names,
            local_label_resolver(class_names),
            n_packets,
            include_sequence_features=include_sequence_features,
            sequence_feature_size_clip=sequence_feature_size_clip,
            node_size_transform=node_size_transform,
            node_size_clip=node_size_clip,
            normalize_node_index=normalize_node_index,
            packet_offset=packet_offset,
        )
        if not records:
            raise RuntimeError(f"Stage II {tool} head found no graphs.")
        labels = [record.label for record in records]
        token_summary = graph_record_token_summary(records, n_packets, class_names)

        train_graphs, val_graphs, test_graphs, base_train_labels, train_pool_labels, test_idx, split_summary = make_splits(
            records,
            labels,
            class_names,
            training_cfg,
            n_packets,
            include_sequence_features,
            sequence_feature_size_clip,
            node_size_transform,
            node_size_clip,
            normalize_node_index,
            packet_offset,
        )
        loss_name, class_weights, training_summary = stage2_training_options(
            stage_cfg,
            train_pool_labels,
            len(class_names),
            base_train_labels,
        )
        split_summary["stage2_head_seed"] = head_seed
        split_summary["stage2_head_seed_offset"] = head_seed_offset
        training_summary["stage2_head_seed"] = head_seed
        training_summary["stage2_head_seed_offset"] = head_seed_offset
        logger.info(
            "Stage II %s head: classes=%d raw_graphs=%d train=%d val=%d test=%d offset=%d n_packets=%d",
            tool,
            len(class_names),
            len(records),
            len(train_graphs),
            len(val_graphs),
            len(test_graphs),
            packet_offset,
            n_packets,
        )
        logger.info(
            "Stage II %s head training: loss=%s checkpoint_metric=%s augmentation=%s augmented=%d",
            tool,
            training_summary["loss"],
            training_summary["checkpoint_metric"],
            split_summary["augmentation"]["enabled"],  # type: ignore[index]
            split_summary["augmentation"]["augmented_graphs"],  # type: ignore[index]
        )
        logger.info(
            "Stage II %s head node_features: size_transform=%s size_clip=%.1f normalize_index=%s",
            tool,
            node_size_transform,
            node_size_clip,
            normalize_node_index,
        )
        if include_sequence_features:
            logger.info("Stage II %s head sequence_global_features dim=%d size_clip=%.1f", tool, global_feature_dim, sequence_feature_size_clip)

        model = Stage2GAT(
            in_channels=2,
            hidden_channels=int(model_cfg["hidden_channels"]),
            num_classes=len(class_names),
            num_heads=int(model_cfg["num_heads"]),
            dropout_rate=float(model_cfg["dropout_rate"]),
            global_feature_dim=global_feature_dim,
        )
        head_dir = output_dir / f"{tool}_head"
        trained = train_graph_classifier(
            model=model,
            train_graphs=train_graphs,
            val_graphs=val_graphs,
            batch_size=int(stage_cfg["batch_size"]),
            epochs=int(stage_cfg["epochs"]),
            lr=float(stage_cfg["lr"]),
            device=device,
            logger=logger,
            ckpt_path=head_dir / "model_best.pth",
            loss_name=loss_name,
            class_weights=class_weights,
            focal_gamma=float(training_summary["focal_gamma"]),
            checkpoint_metric=str(training_summary["checkpoint_metric"]),
            checkpoint_macro_weight=float(training_summary["checkpoint_macro_weight"]),
            num_classes=len(class_names),
            history_path=head_dir / "training_history.json",
        )
        metrics, y_true_local, y_pred_local = evaluate_graph_classifier_with_predictions(
            trained,
            test_graphs=test_graphs,
            class_names=class_names,
            batch_size=int(stage_cfg["batch_size"]),
            device=device,
        )
        head_dir.mkdir(parents=True, exist_ok=True)
        dump_json(head_dir / "metrics.json", metrics)
        if robustness_on:
            entries = class_pcap_entries(config, class_names)
            validate_pcap_entries(records, entries)
            clean_meta = clean_sample_meta(entries, test_idx)
            write_variant_outputs(head_dir, "clean_test", class_names, metrics, y_true_local, y_pred_local, clean_meta)
            local_to_global_clean = [STAGE2_TO_ID[class_name] for class_name in class_names]
            variant_aggregate_true["clean_test"].extend(local_to_global_clean[label] for label in y_true_local)
            variant_aggregate_pred["clean_test"].extend(local_to_global_clean[label] for label in y_pred_local)
            variant_aggregate_meta["clean_test"].extend(clean_meta)
            variant_summaries["clean_test"][tool] = {
                "variant": "clean_test",
                "ratio": 0.0,
                "samples": len(test_idx),
                "changed_samples": 0,
                "unchanged_samples": len(test_idx),
                "notes": ["clean baseline"],
                "token_length_summary": token_summary,
            }
            for variant in robustness_variant_names:
                if variant == "clean_test":
                    continue
                perturbed_graphs, sample_meta, variant_summary = build_stage2_perturbed_graphs(
                    config,
                    records,
                    test_idx,
                    variant,
                    class_names,
                )
                variant_metrics, variant_true, variant_pred = evaluate_graph_classifier_with_predictions(
                    trained,
                    test_graphs=perturbed_graphs,
                    class_names=class_names,
                    batch_size=int(stage_cfg["batch_size"]),
                    device=device,
                )
                write_variant_outputs(head_dir, variant, class_names, variant_metrics, variant_true, variant_pred, sample_meta)
                variant_aggregate_true[variant].extend(local_to_global_clean[label] for label in variant_true)
                variant_aggregate_pred[variant].extend(local_to_global_clean[label] for label in variant_pred)
                variant_aggregate_meta[variant].extend(sample_meta)
                variant_summaries[variant][tool] = variant_summary
                logger.info(
                    "Stage II %s robustness %s acc=%.4f macro_f1=%.4f min_f1=%.4f changed=%d/%d",
                    tool,
                    variant,
                    variant_metrics["accuracy"],
                    variant_metrics["macro_f1"],
                    variant_metrics.get("min_class_f1", 0.0),
                    variant_summary["changed_samples"],
                    variant_summary["samples"],
                )
        local_to_global = [STAGE2_TO_ID[class_name] for class_name in class_names]
        aggregate_true.extend(local_to_global[label] for label in y_true_local)
        aggregate_pred.extend(local_to_global[label] for label in y_pred_local)
        head_metrics[tool] = metrics
        per_head_split[tool] = {
            "raw_counts": raw_counts,
            "token_length_summary": token_summary,
            "training": training_summary,
            **split_summary,
        }

    aggregate_metrics = metrics_from_predictions(aggregate_true, aggregate_pred, STAGE2_CLASSES)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(output_dir / "metrics.json", aggregate_metrics)
    robustness_summary = None
    if robustness_on:
        variant_metrics_by_name: Dict[str, Dict[str, object]] = {}
        for variant in robustness_variant_names:
            metrics_for_variant = metrics_from_predictions(
                variant_aggregate_true[variant],
                variant_aggregate_pred[variant],
                STAGE2_CLASSES,
            )
            write_variant_outputs(
                output_dir,
                variant,
                STAGE2_CLASSES,
                metrics_for_variant,
                variant_aggregate_true[variant],
                variant_aggregate_pred[variant],
                variant_aggregate_meta[variant],
            )
            variant_metrics_by_name[variant] = metrics_for_variant
        robustness_summary = {
            "enabled": True,
            "applies_to": "test_only",
            "train_perturbed": False,
            "ratio": robustness_ratio(config),
            "seed": robustness_seed(config),
            "variants": variant_summaries,
            "deltas_vs_clean_test": metric_deltas(variant_metrics_by_name),
        }
        dump_json(output_dir / "perturbation_summary.json", robustness_summary)
    return aggregate_metrics, output_dir


def run_stage2(config: Dict[str, object], after_dir: Path, run_dir: Path, logger) -> Tuple[Dict[str, object], Path]:
    classifier = stage2_classifier_from_config(config)
    if classifier == "tool_heads":
        return run_stage2_tool_heads(config, after_dir, run_dir, logger)
    if classifier != "single_head":
        raise ValueError(f"Unsupported Stage II classifier: {classifier}")

    txt_map = {class_name: after_dir / f"{class_name}_packets.txt" for class_name in MALICIOUS_CLASSES}
    n_packets = stage2_n_packets_from_config(config)
    packet_offset = stage2_packet_offset_from_config(config)
    include_sequence_features, sequence_feature_size_clip, global_feature_dim = sequence_global_feature_config(config)
    node_size_transform, node_size_clip, normalize_node_index = graph_node_feature_config(config)
    graphs, labels, raw_counts = collect_graphs(
        txt_map,
        MALICIOUS_CLASSES,
        raw_to_stage2_id,
        n_packets,
        include_sequence_features=include_sequence_features,
        sequence_feature_size_clip=sequence_feature_size_clip,
        node_size_transform=node_size_transform,
        node_size_clip=node_size_clip,
        normalize_node_index=normalize_node_index,
        packet_offset=packet_offset,
    )
    if not graphs:
        raise RuntimeError("Stage II found no graphs.")

    training_cfg = config["training"]  # type: ignore[index]
    split_cfg = training_cfg["split"]
    seed = int(training_cfg["seed"])
    train_idx, test_idx = safe_stratified_split(labels, float(split_cfg["test_size"]), seed)
    val_size = float(split_cfg.get("val_size", 0.0))
    val_idx = []
    if val_size > 0.0 and train_idx:
        inner_train, inner_val = safe_stratified_split([labels[idx] for idx in train_idx], val_size, seed + 1)
        val_idx = [train_idx[idx] for idx in inner_val]
        train_idx = [train_idx[idx] for idx in inner_train]

    oversampling_cfg = training_cfg["oversampling"]
    train_pool_labels = [labels[idx] for idx in train_idx]
    if bool(oversampling_cfg["enabled"]):
        sampled = oversample_indices(train_pool_labels)
        train_graphs = [graphs[train_idx[idx]] for idx in sampled]
    else:
        train_graphs = index_subset(graphs, train_idx)

    val_graphs = index_subset(graphs, val_idx)
    test_graphs = index_subset(graphs, test_idx)

    stage_cfg = training_cfg["stage2"]
    loss_name, class_weights, training_summary = stage2_training_options(
        stage_cfg,
        train_pool_labels,
        len(STAGE2_CLASSES),
    )
    model_cfg = config["model"]["stage2"]  # type: ignore[index]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Stage2GAT(
        in_channels=2,
        hidden_channels=int(model_cfg["hidden_channels"]),
        num_classes=len(STAGE2_CLASSES),
        num_heads=int(model_cfg["num_heads"]),
        dropout_rate=float(model_cfg["dropout_rate"]),
        global_feature_dim=global_feature_dim,
    )

    output_dir = run_dir / "stage2"
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
        loss_name=loss_name,
        class_weights=class_weights,
        focal_gamma=float(training_summary["focal_gamma"]),
        checkpoint_metric=str(training_summary["checkpoint_metric"]),
        checkpoint_macro_weight=float(training_summary["checkpoint_macro_weight"]),
        num_classes=len(STAGE2_CLASSES),
        history_path=output_dir / "training_history.json",
    )
    metrics = evaluate_graph_classifier(
        trained,
        test_graphs=test_graphs,
        class_names=STAGE2_CLASSES,
        batch_size=int(stage_cfg["batch_size"]),
        device=device,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(output_dir / "metrics.json", metrics)
    return metrics, output_dir
