from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from v2_common import (
    GraphRecord,
    dump_json,
    graph_data_from_line,
    resolve_path,
    sequence_slice_lengths,
)
from v2_preprocess import (
    PacketRecord,
    awm_source_window_from_config,
    cache_n_packets_from_config,
    encode_flow_tokens,
    labels_from_components,
    list_class_pcaps,
    max_samples_for_class,
    precompute_components,
    read_flow_from_pcap,
    run_awm_kws_on_flow,
    slice_awm_source_tokens,
)


def robustness_config(config: Dict[str, object]) -> Dict[str, object]:
    raw = config.get("robustness", {})
    return raw if isinstance(raw, dict) else {}


def robustness_enabled(config: Dict[str, object]) -> bool:
    return bool(robustness_config(config).get("enabled", False))


def robustness_variants(config: Dict[str, object]) -> List[str]:
    cfg = robustness_config(config)
    variants = cfg.get("variants", ["clean_test", "combined7p5_test"])
    if isinstance(variants, str):
        return [variants]
    return [str(item) for item in variants]


def robustness_ratio(config: Dict[str, object]) -> float:
    return float(robustness_config(config).get("ratio", 0.075))


def robustness_seed(config: Dict[str, object]) -> int:
    cfg = robustness_config(config)
    if "seed" in cfg:
        return int(cfg["seed"])
    training_cfg = config.get("training", {})
    if isinstance(training_cfg, dict):
        return int(training_cfg.get("seed", 42))
    return 42


def summarize_numbers(values: Sequence[float], full_window: Optional[int] = None) -> Dict[str, object]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "full_window_rate": None}
    arr = np.array(values, dtype=float)
    summary: Dict[str, object] = {
        "count": int(len(values)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
    }
    if full_window is not None and full_window > 0:
        summary["full_window_rate"] = float(np.mean(arr >= float(full_window)))
    else:
        summary["full_window_rate"] = None
    return summary


def class_pcap_entries(config: Dict[str, object], class_names: Sequence[str]) -> List[Tuple[str, Path]]:
    data_dir = resolve_path(config["paths"]["data_dir"])  # type: ignore[index]
    entries: List[Tuple[str, Path]] = []
    for class_name in class_names:
        limit = max_samples_for_class(config, class_name)
        for path in list_class_pcaps(data_dir, class_name, limit):
            entries.append((class_name, path))
    return entries


def validate_pcap_entries(records: Sequence[GraphRecord], entries: Sequence[Tuple[str, Path]]) -> None:
    if len(records) != len(entries):
        raise RuntimeError(
            f"Robustness pcap mapping mismatch: records={len(records)} pcaps={len(entries)}. "
            "This evaluator requires cache written_lines to match input_files."
        )
    for idx, (record, (class_name, _)) in enumerate(zip(records, entries)):
        if record.class_name != class_name:
            raise RuntimeError(
                f"Robustness pcap mapping mismatch at index {idx}: "
                f"record={record.class_name} pcap_class={class_name}"
            )


def clone_packet(pkt: PacketRecord) -> PacketRecord:
    return PacketRecord(pkt.ts, pkt.src, pkt.sport, pkt.dst, pkt.dport, pkt.udp_plen)


def clone_flow(flow: Sequence[PacketRecord]) -> List[PacketRecord]:
    return [clone_packet(pkt) for pkt in flow]


def apply_jitter(flow: Sequence[PacketRecord], ratio: float, rng: random.Random) -> List[PacketRecord]:
    out = clone_flow(flow)
    if len(out) < 2:
        return out
    original_ts = [pkt.ts for pkt in out]
    out[0].ts = original_ts[0]
    current = out[0].ts
    for idx in range(1, len(out)):
        iat = max(0.0, original_ts[idx] - original_ts[idx - 1])
        current += max(0.0, iat * (1.0 + rng.uniform(-ratio, ratio)))
        out[idx].ts = current
    return out


def apply_loss(flow: Sequence[PacketRecord], ratio: float, rng: random.Random) -> List[PacketRecord]:
    kept = [clone_packet(pkt) for pkt in flow if rng.random() >= ratio]
    if not kept and flow:
        kept = [clone_packet(flow[0])]
    return kept


def apply_retrans(flow: Sequence[PacketRecord], ratio: float, rng: random.Random) -> List[PacketRecord]:
    out = clone_flow(flow)
    if not out:
        return out
    duplicate_count = int(round(len(out) * ratio))
    for _ in range(max(0, duplicate_count)):
        src_idx = rng.randrange(len(out))
        insert_idx = min(src_idx + 1, len(out))
        pkt = clone_packet(out[src_idx])
        if insert_idx < len(out):
            next_ts = out[insert_idx].ts
            pkt.ts = (out[src_idx].ts + next_ts) / 2.0 if next_ts > out[src_idx].ts else out[src_idx].ts + 1e-9
        else:
            pkt.ts = out[src_idx].ts + 1e-9
        out.insert(insert_idx, pkt)
    return out


def perturb_flow(
    flow: Sequence[PacketRecord],
    variant: str,
    ratio: float,
    seed: int,
    dataset_index: int,
) -> Tuple[List[PacketRecord], bool, str]:
    original = clone_flow(flow)
    if variant == "clean_test":
        return original, False, "clean baseline"
    offsets = {
        "jitter7p5_test": 10015,
        "loss7p5_test": 20015,
        "retrans7p5_test": 30015,
        "combined7p5_test": 40015,
        "combined5_test": 40017,
        "combined2p5_test": 40018,
    }
    rng = random.Random(int(seed) + int(dataset_index) * 1009 + offsets.get(variant, 0))
    if variant == "jitter7p5_test":
        out = apply_jitter(original, ratio, rng)
        return out, bool(len(out) > 1), f"IAT jitter +/-{ratio:.1%}"
    if variant == "loss7p5_test":
        out = apply_loss(original, ratio, rng)
        return out, len(out) != len(original), f"drop {ratio:.1%} packets without refill"
    if variant == "retrans7p5_test":
        out = apply_retrans(original, ratio, rng)
        return out, len(out) != len(original), f"duplicate {ratio:.1%} packets near original position"
    if variant in {"combined7p5_test", "combined5_test", "combined2p5_test"}:
        out = apply_retrans(apply_loss(apply_jitter(original, ratio, rng), ratio, rng), ratio, rng)
        return out, True, f"jitter +/-{ratio:.1%}, then loss {ratio:.1%}, then retrans {ratio:.1%}"
    raise ValueError(f"Unknown robustness variant: {variant}")


def encode_flow_line(flow: Sequence[PacketRecord], heartbeat_cfg: Dict[str, object]) -> Optional[str]:
    if not flow:
        return None
    alpha = float(heartbeat_cfg["weights"]["alpha"])  # type: ignore[index]
    beta = float(heartbeat_cfg["weights"]["beta"])  # type: ignore[index]
    gamma = float(heartbeat_cfg["weights"]["gamma"])  # type: ignore[index]
    hb_percentile = float(heartbeat_cfg["percentile_threshold"])
    mutable_flow = clone_flow(flow)
    components = precompute_components(mutable_flow, heartbeat_cfg)
    labels = labels_from_components(components, len(mutable_flow), alpha, beta, gamma, hb_percentile)
    return encode_flow_tokens(mutable_flow, labels)


def build_perturbed_line_from_pcap(
    config: Dict[str, object],
    pcap_path: Path,
    variant: str,
    ratio: float,
    seed: int,
    dataset_index: int,
) -> Tuple[str, Dict[str, object]]:
    cache_n_packets = cache_n_packets_from_config(config)
    flow = read_flow_from_pcap(str(pcap_path))
    original_flow_len = len(flow)
    flow = flow[:cache_n_packets] if cache_n_packets > 0 else flow
    perturbed, changed, note = perturb_flow(flow, variant, ratio, seed, dataset_index)
    line = encode_flow_line(perturbed, config["heartbeat"])  # type: ignore[index]
    if line is None:
        raise RuntimeError(f"Perturbed flow produced no tokens: {pcap_path}")
    stats = {
        "original_udp_packets": int(original_flow_len),
        "clean_cache_packets": int(len(flow)),
        "perturbed_packets": int(len(perturbed)),
        "changed": bool(changed),
        "note": note,
        "before_tokens": int(len(line.split())),
    }
    return line, stats


def stage1_graph_from_line(config: Dict[str, object], line: str) -> Tuple[Data, int]:
    from v2_stage1 import stage1_n_packets_from_config, stage1_packet_offset_from_config

    n_packets = stage1_n_packets_from_config(config)
    packet_offset = stage1_packet_offset_from_config(config)
    graph = graph_data_from_line(
        line,
        n_packets=n_packets,
        packet_offset=packet_offset,
    )
    if graph is None:
        raise RuntimeError("Stage I robustness graph is empty.")
    _, sliced_len = sequence_slice_lengths(line, n_packets, packet_offset)
    return graph, int(sliced_len)


def stage2_graph_from_line(config: Dict[str, object], line: str) -> Tuple[Data, Dict[str, int]]:
    from v2_stage2 import graph_node_feature_config, sequence_global_feature_config, stage2_n_packets_from_config, stage2_packet_offset_from_config

    tokens = [token for token in line.split() if token]
    source_window = awm_source_window_from_config(config)
    source_tokens = slice_awm_source_tokens(tokens, source_window)
    picked = run_awm_kws_on_flow(source_tokens, config["awm_kws"])  # type: ignore[index]
    after_line = " ".join(source_tokens[idx] for idx in picked)
    n_packets = stage2_n_packets_from_config(config)
    packet_offset = stage2_packet_offset_from_config(config)
    include_sequence_features, sequence_feature_size_clip, _ = sequence_global_feature_config(config)
    node_size_transform, node_size_clip, normalize_node_index = graph_node_feature_config(config)
    graph = graph_data_from_line(
        after_line,
        n_packets=n_packets,
        include_sequence_features=include_sequence_features,
        sequence_feature_size_clip=sequence_feature_size_clip,
        node_size_transform=node_size_transform,
        node_size_clip=node_size_clip,
        normalize_node_index=normalize_node_index,
        packet_offset=packet_offset,
    )
    if graph is None:
        raise RuntimeError("Stage II robustness graph is empty.")
    _, final_sliced = sequence_slice_lengths(after_line, n_packets, packet_offset)
    return graph, {
        "awm_source_tokens": int(len(source_tokens)),
        "after_awm_tokens": int(len(picked)),
        "final_sliced_tokens": int(final_sliced),
    }


def evaluate_with_predictions(
    model: torch.nn.Module,
    graphs: Sequence[Data],
    class_names: Sequence[str],
    batch_size: int,
    device: torch.device,
) -> Tuple[Dict[str, object], List[int], List[int]]:
    if not graphs:
        raise RuntimeError("No graphs available for robustness evaluation.")
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    loader = DataLoader(list(graphs), batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"))
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            y_pred.extend(out.argmax(dim=1).cpu().numpy().tolist())
            y_true.extend(batch.y.cpu().numpy().tolist())
    return metrics_from_predictions(y_true, y_pred, class_names), y_true, y_pred


def metrics_from_predictions(y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]) -> Dict[str, object]:
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


def report_lines(class_names: Sequence[str], metrics: Dict[str, object]) -> str:
    lines = [
        f"Accuracy (overall): {metrics['accuracy'] * 100:.2f}%",
        f"Macro Precision: {metrics['macro_precision'] * 100:.2f}%",
        f"Macro Recall: {metrics['macro_recall'] * 100:.2f}%",
        f"Macro F1: {metrics['macro_f1'] * 100:.2f}%",
        f"Min class F1: {metrics.get('min_class_f1', 0.0) * 100:.2f}%",
        "",
        "Per-class metrics:",
    ]
    class_metrics: Dict[str, Dict[str, object]] = metrics["class_metrics"]  # type: ignore[assignment]
    for class_name in class_names:
        item = class_metrics[class_name]
        lines.append(
            f"{class_name}: P={item['precision']:.4f} R={item['recall']:.4f} "
            f"F1={item['f1']:.4f} N={item['support']}"
        )
    return "\n".join(lines) + "\n"


def write_variant_outputs(
    output_dir: Path,
    variant: str,
    class_names: Sequence[str],
    metrics: Dict[str, object],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    sample_meta: Sequence[Dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(output_dir / f"metrics_{variant}.json", metrics)
    (output_dir / f"report_{variant}.txt").write_text(report_lines(class_names, metrics), encoding="utf-8")

    cm = np.array(metrics["confusion_matrix"], dtype=int)
    with (output_dir / f"confusion_matrix_{variant}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *class_names])
        for idx, class_name in enumerate(class_names):
            writer.writerow([class_name, *cm[idx].tolist()])

    with (output_dir / f"misclassified_samples_{variant}.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["sample_order", "dataset_index", "pcap", "raw_class", "true_label", "pred_label"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for order, (true_id, pred_id) in enumerate(zip(y_true, y_pred)):
            if int(true_id) == int(pred_id):
                continue
            meta = sample_meta[order] if order < len(sample_meta) else {}
            writer.writerow({
                "sample_order": order,
                "dataset_index": meta.get("dataset_index", ""),
                "pcap": meta.get("pcap", ""),
                "raw_class": meta.get("raw_class", ""),
                "true_label": class_names[int(true_id)],
                "pred_label": class_names[int(pred_id)],
            })


def clean_sample_meta(
    entries: Sequence[Tuple[str, Path]],
    test_indices: Sequence[int],
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for idx in test_indices:
        class_name, pcap_path = entries[idx]
        out.append({
            "dataset_index": int(idx),
            "pcap": str(pcap_path),
            "raw_class": class_name,
        })
    return out


def build_stage1_perturbed_graphs(
    config: Dict[str, object],
    records: Sequence[GraphRecord],
    test_indices: Sequence[int],
    variant: str,
    class_names: Sequence[str],
) -> Tuple[List[Data], List[Dict[str, object]], Dict[str, object]]:
    entries = class_pcap_entries(config, class_names)
    validate_pcap_entries(records, entries)
    ratio = robustness_ratio(config)
    seed = robustness_seed(config)
    graphs: List[Data] = []
    sample_meta: List[Dict[str, object]] = []
    before_packets: List[int] = []
    after_packets: List[int] = []
    before_tokens: List[int] = []
    final_tokens: List[int] = []
    changed = 0
    notes: List[str] = []
    for idx in test_indices:
        record = records[idx]
        class_name, pcap_path = entries[idx]
        line, stats = build_perturbed_line_from_pcap(config, pcap_path, variant, ratio, seed, idx)
        graph, sliced_len = stage1_graph_from_line(config, line)
        graph.y = torch.tensor([record.label], dtype=torch.long)
        graphs.append(graph)
        sample_meta.append({"dataset_index": int(idx), "pcap": str(pcap_path), "raw_class": class_name})
        before_packets.append(int(stats["clean_cache_packets"]))
        after_packets.append(int(stats["perturbed_packets"]))
        before_tokens.append(int(stats["before_tokens"]))
        final_tokens.append(int(sliced_len))
        if stats["changed"]:
            changed += 1
        if str(stats["note"]) not in notes:
            notes.append(str(stats["note"]))
    summary = {
        "variant": variant,
        "ratio": ratio if variant != "clean_test" else 0.0,
        "samples": int(len(test_indices)),
        "changed_samples": int(changed),
        "unchanged_samples": int(len(test_indices) - changed),
        "notes": notes,
        "clean_cache_packet_count_stats": summarize_numbers(before_packets, cache_n_packets_from_config(config)),
        "perturbed_packet_count_stats": summarize_numbers(after_packets),
        "before_token_count_stats": summarize_numbers(before_tokens, cache_n_packets_from_config(config)),
        "final_sliced_token_count_stats": summarize_numbers(final_tokens),
    }
    return graphs, sample_meta, summary


def build_stage2_perturbed_graphs(
    config: Dict[str, object],
    records: Sequence[GraphRecord],
    test_indices: Sequence[int],
    variant: str,
    class_names: Sequence[str],
) -> Tuple[List[Data], List[Dict[str, object]], Dict[str, object]]:
    entries = class_pcap_entries(config, class_names)
    validate_pcap_entries(records, entries)
    ratio = robustness_ratio(config)
    seed = robustness_seed(config)
    graphs: List[Data] = []
    sample_meta: List[Dict[str, object]] = []
    before_packets: List[int] = []
    after_packets: List[int] = []
    before_tokens: List[int] = []
    awm_source_tokens: List[int] = []
    after_awm_tokens: List[int] = []
    final_tokens: List[int] = []
    changed = 0
    notes: List[str] = []
    for idx in test_indices:
        record = records[idx]
        class_name, pcap_path = entries[idx]
        line, stats = build_perturbed_line_from_pcap(config, pcap_path, variant, ratio, seed, idx)
        graph, graph_stats = stage2_graph_from_line(config, line)
        graph.y = torch.tensor([record.label], dtype=torch.long)
        graphs.append(graph)
        sample_meta.append({"dataset_index": int(idx), "pcap": str(pcap_path), "raw_class": class_name})
        before_packets.append(int(stats["clean_cache_packets"]))
        after_packets.append(int(stats["perturbed_packets"]))
        before_tokens.append(int(stats["before_tokens"]))
        awm_source_tokens.append(int(graph_stats["awm_source_tokens"]))
        after_awm_tokens.append(int(graph_stats["after_awm_tokens"]))
        final_tokens.append(int(graph_stats["final_sliced_tokens"]))
        if stats["changed"]:
            changed += 1
        if str(stats["note"]) not in notes:
            notes.append(str(stats["note"]))
    summary = {
        "variant": variant,
        "ratio": ratio if variant != "clean_test" else 0.0,
        "samples": int(len(test_indices)),
        "changed_samples": int(changed),
        "unchanged_samples": int(len(test_indices) - changed),
        "notes": notes,
        "clean_cache_packet_count_stats": summarize_numbers(before_packets, cache_n_packets_from_config(config)),
        "perturbed_packet_count_stats": summarize_numbers(after_packets),
        "before_token_count_stats": summarize_numbers(before_tokens, cache_n_packets_from_config(config)),
        "awm_source_token_count_stats": summarize_numbers(
            awm_source_tokens,
            int(awm_source_window_from_config(config).get("n_packets", 0)),
        ),
        "after_awm_token_count_stats": summarize_numbers(after_awm_tokens),
        "final_sliced_token_count_stats": summarize_numbers(final_tokens),
    }
    return graphs, sample_meta, summary


def metric_deltas(metrics_by_variant: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    clean = metrics_by_variant.get("clean_test")
    if clean is None:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for variant, metrics in metrics_by_variant.items():
        out[variant] = {
            "delta_accuracy": float(metrics["accuracy"]) - float(clean["accuracy"]),
            "delta_macro_f1": float(metrics["macro_f1"]) - float(clean["macro_f1"]),
            "delta_min_class_f1": float(metrics.get("min_class_f1", 0.0)) - float(clean.get("min_class_f1", 0.0)),
        }
    return out
