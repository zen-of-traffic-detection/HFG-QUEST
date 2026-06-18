from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_GLOBAL_FEATURE_DIM = 32


@dataclass(frozen=True)
class GraphRecord:
    graph: Data
    label: int
    class_name: str
    line: str
    raw_token_count: int = 0
    sliced_token_count: int = 0


def load_yaml_or_json(path: Union[str, os.PathLike[str]]) -> Dict[str, object]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load YAML configuration files.") from exc
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must decode to a mapping.")
    return data


def dump_json(path: Union[str, os.PathLike[str]], data: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def deep_merge(base: Dict[str, object], override: Dict[str, object]) -> Dict[str, object]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def freeze_for_hash(value: object) -> object:
    if isinstance(value, dict):
        return {k: freeze_for_hash(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [freeze_for_hash(item) for item in value]
    return value


def short_hash(payload: object, length: int = 10) -> str:
    normalized = json.dumps(freeze_for_hash(payload), sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:length]


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Union[str, os.PathLike[str]]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(value: Union[str, os.PathLike[str]]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_logger(log_path: Union[str, os.PathLike[str]]) -> logging.Logger:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"hfg_v2_{log_path.stem}_{short_hash(str(log_path), 6)}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def parse_line_to_seq(line: str) -> List[Tuple[int, bool]]:
    seq: List[Tuple[int, bool]] = []
    for token in line.split():
        is_hb = token.startswith("+")
        core = token[1:] if is_hb else token
        seq.append((int(core), is_hb))
    return seq


def sequence_global_features(sequence: Sequence[Tuple[int, bool]], max_packets: int, size_clip: float) -> List[float]:
    if not sequence:
        return [0.0] * SEQUENCE_GLOBAL_FEATURE_DIM

    values = np.array([float(item[0]) for item in sequence], dtype=np.float32)
    abs_values = np.abs(values)
    hb = np.array([1.0 if item[1] else 0.0 for item in sequence], dtype=np.float32)
    signs = np.sign(values)
    n = float(len(sequence))
    clip = float(size_clip) if size_clip > 0 else 1500.0

    pos = (values > 0).astype(np.float32)
    neg = (values < 0).astype(np.float32)
    zero = (values == 0).astype(np.float32)
    sign_switches = np.mean(signs[1:] != signs[:-1]) if len(signs) > 1 else 0.0
    abs_sum = float(np.sum(abs_values)) + 1e-6

    hb_segments = 0
    hb_run_lengths: List[int] = []
    current_run = 0
    for is_hb in hb:
        if is_hb:
            if current_run == 0:
                hb_segments += 1
            current_run += 1
        elif current_run:
            hb_run_lengths.append(current_run)
            current_run = 0
    if current_run:
        hb_run_lengths.append(current_run)
    hb_mean_run = float(np.mean(hb_run_lengths)) if hb_run_lengths else 0.0

    def clipped(value: float) -> float:
        return float(np.clip(value / clip, 0.0, 1.0))

    features: List[float] = [
        float(len(sequence) / max(float(max_packets), 1.0)),
        float(np.mean(hb)),
        float(hb_segments / n),
        float(hb_mean_run / n),
        float(np.mean(pos)),
        float(np.mean(neg)),
        float(np.mean(zero)),
        float(sign_switches),
        float(np.sum(abs_values * pos) / abs_sum),
        float(np.sum(abs_values * neg) / abs_sum),
        clipped(float(np.mean(abs_values))),
        clipped(float(np.std(abs_values))),
        clipped(float(np.min(abs_values))),
        clipped(float(np.percentile(abs_values, 25))),
        clipped(float(np.percentile(abs_values, 50))),
        clipped(float(np.percentile(abs_values, 75))),
        clipped(float(np.percentile(abs_values, 90))),
        clipped(float(np.max(abs_values))),
        float(np.mean(abs_values <= 64.0)),
        float(np.mean(abs_values <= 512.0)),
        float(np.mean(abs_values >= 1200.0)),
        float(np.clip(np.mean(values) / clip, -1.0, 1.0)),
        clipped(float(np.std(values))),
    ]

    chunks = np.array_split(np.arange(len(sequence)), 3)
    for chunk in chunks:
        if len(chunk) == 0:
            features.extend([0.0, 0.0, 0.0])
            continue
        features.extend([
            float(np.mean(pos[chunk])),
            float(np.mean(hb[chunk])),
            clipped(float(np.mean(abs_values[chunk]))),
        ])

    if len(features) != SEQUENCE_GLOBAL_FEATURE_DIM:
        raise RuntimeError(f"Expected {SEQUENCE_GLOBAL_FEATURE_DIM} sequence features, got {len(features)}.")
    return features


def build_graph_custom(sequence: Sequence[Tuple[int, bool]]) -> nx.Graph:
    if not sequence:
        return nx.Graph()

    groups: List[List[int]] = []
    sub_list: List[int] = []
    is_new_node = False

    for item in sequence[:-1]:
        value = int(item[0])
        hb = bool(item[1])
        if hb and not is_new_node:
            sub_list.append(value)
        elif hb and is_new_node:
            groups.append(sub_list)
            sub_list = [value]
            is_new_node = False
        else:
            sub_list.append(value)
            is_new_node = True

    sub_list.append(int(sequence[-1][0]))
    groups.append(sub_list)

    graph = nx.Graph()
    head: Optional[Tuple[int, int]] = None
    counter = 0

    for edge in groups:
        if len(edge) < 2:
            continue
        current_head = (edge[0], counter)
        previous = (edge[0], counter)
        counter += 1
        for idx in range(len(edge) - 1):
            nxt = (edge[idx + 1], counter)
            graph.add_edge(previous, nxt)
            previous = nxt
            counter += 1
        if head is None:
            head = current_head
        else:
            graph.add_edge(head, current_head)
            head = current_head
    return graph


def graph_data_from_line(
    line: str,
    n_packets: int,
    include_sequence_features: bool = False,
    sequence_feature_size_clip: float = 1500.0,
    node_size_transform: str = "raw",
    node_size_clip: float = 1500.0,
    normalize_node_index: bool = False,
    packet_offset: int = 0,
) -> Optional[Data]:
    seq = parse_line_to_seq(line)
    seq = slice_sequence(seq, n_packets, packet_offset)
    graph = build_graph_custom(seq)
    nodes = list(graph.nodes())
    if not nodes:
        return None
    transform = str(node_size_transform).lower()
    clip = float(node_size_clip) if node_size_clip and node_size_clip > 0 else 1500.0

    def packet_size_feature(value: int) -> float:
        if transform in ("raw", "none"):
            return float(value)
        if transform in ("log_signed", "signed_log"):
            sign = 1.0 if value >= 0 else -1.0
            magnitude = min(abs(float(value)), clip)
            return float(sign * np.log1p(magnitude) / np.log1p(clip))
        if transform == "linear_clip":
            return float(np.clip(float(value) / clip, -1.0, 1.0))
        raise ValueError(f"Unsupported node_size_transform: {node_size_transform}")

    index_denominator = max(float(n_packets), 1.0)
    x = torch.tensor(
        np.array(
            [
                [
                    packet_size_feature(int(node[0])),
                    float(node[1]) / index_denominator if normalize_node_index else float(node[1]),
                ]
                for node in nodes
            ],
            dtype=np.float32,
        ),
        dtype=torch.float32,
    )
    adjacency = nx.to_numpy_array(graph, dtype=np.float32)
    rows, cols = np.nonzero(adjacency)
    edge_index = torch.tensor(np.vstack([rows, cols]).astype(np.int64), dtype=torch.long).contiguous()
    data = Data(x=x, edge_index=edge_index)
    if include_sequence_features:
        features = sequence_global_features(seq, n_packets, sequence_feature_size_clip)
        data.global_x = torch.tensor([features], dtype=torch.float32)
    return data


def slice_sequence(
    sequence: Sequence[Tuple[int, bool]],
    n_packets: int,
    packet_offset: int = 0,
) -> List[Tuple[int, bool]]:
    offset = max(0, int(packet_offset))
    sliced = list(sequence[offset:])
    if n_packets > 0:
        sliced = sliced[:int(n_packets)]
    return sliced


def sequence_slice_lengths(line: str, n_packets: int, packet_offset: int = 0) -> Tuple[int, int]:
    sequence = parse_line_to_seq(line)
    return len(sequence), len(slice_sequence(sequence, n_packets, packet_offset))


def summarize_token_lengths(values: Sequence[int]) -> Dict[str, object]:
    if not values:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
        }
    array = np.array([int(value) for value in values], dtype=np.float64)
    return {
        "count": int(array.size),
        "min": int(np.min(array)),
        "max": int(np.max(array)),
        "mean": float(np.mean(array)),
        "p25": float(np.percentile(array, 25)),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
    }


def graph_record_token_summary(
    records: Sequence[GraphRecord],
    n_packets: int,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    raw_lengths = [record.raw_token_count for record in records]
    sliced_lengths = [record.sliced_token_count for record in records]
    summary: Dict[str, object] = {
        "raw_tokens": summarize_token_lengths(raw_lengths),
        "sliced_tokens": summarize_token_lengths(sliced_lengths),
        "full_window_rate": 0.0,
    }
    if records and n_packets > 0:
        summary["full_window_rate"] = float(np.mean([length >= n_packets for length in sliced_lengths]))
    if class_names is not None:
        by_class: Dict[str, object] = {}
        for class_name in class_names:
            class_records = [record for record in records if record.class_name == class_name]
            class_sliced = [record.sliced_token_count for record in class_records]
            class_summary = summarize_token_lengths(class_sliced)
            if class_records and n_packets > 0:
                class_summary["full_window_rate"] = float(np.mean([length >= n_packets for length in class_sliced]))
            else:
                class_summary["full_window_rate"] = 0.0
            by_class[class_name] = class_summary
        summary["by_class"] = by_class
    return summary


def sequence_to_line(sequence: Sequence[Tuple[int, bool]]) -> str:
    return " ".join((f"+{int(value)}" if is_hb else str(int(value))) for value, is_hb in sequence)


def augment_sequence_line(
    line: str,
    rng: random.Random,
    augmentation_cfg: Dict[str, object],
    max_packets: int,
) -> str:
    sequence = parse_line_to_seq(line)
    if max_packets > 0:
        sequence = sequence[:max_packets]
    if len(sequence) < 2:
        return sequence_to_line(sequence)

    min_tokens = int(augmentation_cfg.get("min_tokens", 8))
    size_jitter = float(augmentation_cfg.get("size_jitter", 0.0))
    dropout_prob = float(augmentation_cfg.get("dropout_prob", 0.0))
    duplicate_prob = float(augmentation_cfg.get("duplicate_prob", 0.0))
    crop_max_ratio = float(augmentation_cfg.get("crop_max_ratio", 0.0))
    min_tokens = max(2, min(min_tokens, len(sequence)))

    if crop_max_ratio > 0.0 and len(sequence) > min_tokens:
        max_crop = min(len(sequence) - min_tokens, int(round(len(sequence) * crop_max_ratio)))
        if max_crop > 0:
            left = rng.randint(0, max_crop)
            right = rng.randint(0, max_crop - left)
            if left or right:
                sequence = sequence[left:len(sequence) - right if right else len(sequence)]

    if dropout_prob > 0.0 and len(sequence) > min_tokens:
        dropped = [item for item in sequence if rng.random() >= dropout_prob]
        if len(dropped) >= min_tokens:
            sequence = dropped

    if duplicate_prob > 0.0:
        duplicated: List[Tuple[int, bool]] = []
        for item in sequence:
            duplicated.append(item)
            if rng.random() < duplicate_prob:
                duplicated.append(item)
        sequence = duplicated

    jittered: List[Tuple[int, bool]] = []
    for value, is_hb in sequence:
        new_value = int(value)
        if size_jitter > 0.0 and new_value != 0:
            sign = 1 if new_value > 0 else -1
            magnitude = max(1, abs(new_value))
            factor = rng.uniform(max(0.0, 1.0 - size_jitter), 1.0 + size_jitter)
            new_value = sign * max(1, int(round(magnitude * factor)))
        jittered.append((new_value, is_hb))

    if max_packets > 0:
        jittered = jittered[:max_packets]
    return sequence_to_line(jittered)


def safe_stratified_split(
    labels: Sequence[int],
    test_size: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    indices = list(range(len(labels)))
    if not indices:
        return [], []
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or np.min(counts) < 2 or test_size <= 0.0:
        if len(indices) == 1:
            return indices, []
        split_at = max(1, int(round(len(indices) * (1.0 - test_size))))
        return indices[:split_at], indices[split_at:]
    if test_size < 1.0:
        requested_test = int(np.ceil(len(indices) * test_size))
    else:
        requested_test = int(test_size)
    if requested_test < len(unique) or len(indices) - requested_test < len(unique):
        split_at = max(1, min(len(indices) - 1, int(round(len(indices) * (1.0 - test_size)))))
        return indices[:split_at], indices[split_at:]
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=list(labels),
    )
    return list(train_idx), list(test_idx)


def oversample_indices(labels: Sequence[int]) -> List[int]:
    grouped: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        grouped.setdefault(int(label), []).append(idx)
    if not grouped:
        return []
    max_count = max(len(items) for items in grouped.values())
    sampled: List[int] = []
    for label in sorted(grouped):
        source = grouped[label]
        repeats = max_count // len(source)
        remainder = max_count % len(source)
        sampled.extend(source * repeats)
        sampled.extend(source[:remainder])
    random.shuffle(sampled)
    return sampled


def collect_graphs(
    txt_map: Dict[str, Path],
    class_names: Sequence[str],
    label_resolver,
    n_packets: int,
    include_sequence_features: bool = False,
    sequence_feature_size_clip: float = 1500.0,
    node_size_transform: str = "raw",
    node_size_clip: float = 1500.0,
    normalize_node_index: bool = False,
    packet_offset: int = 0,
) -> Tuple[List[Data], List[int], Dict[str, int]]:
    graphs: List[Data] = []
    labels: List[int] = []
    counts: Dict[str, int] = {}
    for class_name in class_names:
        file_path = txt_map[class_name]
        counts[class_name] = 0
        if not file_path.exists():
            continue
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                graph = graph_data_from_line(
                    stripped,
                    n_packets=n_packets,
                    include_sequence_features=include_sequence_features,
                    sequence_feature_size_clip=sequence_feature_size_clip,
                    node_size_transform=node_size_transform,
                    node_size_clip=node_size_clip,
                    normalize_node_index=normalize_node_index,
                    packet_offset=packet_offset,
                )
                if graph is None:
                    continue
                label = label_resolver(class_name)
                graph.y = torch.tensor([label], dtype=torch.long)
                graphs.append(graph)
                labels.append(label)
                counts[class_name] += 1
    return graphs, labels, counts


def collect_graph_records(
    txt_map: Dict[str, Path],
    class_names: Sequence[str],
    label_resolver,
    n_packets: int,
    include_sequence_features: bool = False,
    sequence_feature_size_clip: float = 1500.0,
    node_size_transform: str = "raw",
    node_size_clip: float = 1500.0,
    normalize_node_index: bool = False,
    packet_offset: int = 0,
) -> Tuple[List[GraphRecord], Dict[str, int]]:
    records: List[GraphRecord] = []
    counts: Dict[str, int] = {}
    for class_name in class_names:
        file_path = txt_map[class_name]
        counts[class_name] = 0
        if not file_path.exists():
            continue
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                graph = graph_data_from_line(
                    stripped,
                    n_packets=n_packets,
                    include_sequence_features=include_sequence_features,
                    sequence_feature_size_clip=sequence_feature_size_clip,
                    node_size_transform=node_size_transform,
                    node_size_clip=node_size_clip,
                    normalize_node_index=normalize_node_index,
                    packet_offset=packet_offset,
                )
                if graph is None:
                    continue
                label = label_resolver(class_name)
                graph.y = torch.tensor([label], dtype=torch.long)
                raw_len, sliced_len = sequence_slice_lengths(stripped, n_packets, packet_offset)
                records.append(GraphRecord(
                    graph=graph,
                    label=label,
                    class_name=class_name,
                    line=stripped,
                    raw_token_count=raw_len,
                    sliced_token_count=sliced_len,
                ))
                counts[class_name] += 1
    return records, counts


def index_subset(items: Sequence[Data], indices: Sequence[int]) -> List[Data]:
    return [items[idx] for idx in indices]


def label_count_dict(labels: Sequence[int], class_names: Sequence[str]) -> Dict[str, int]:
    counts = {class_name: 0 for class_name in class_names}
    for label in labels:
        counts[class_names[int(label)]] += 1
    return counts


def graph_label_count_dict(graphs: Sequence[Data], class_names: Sequence[str]) -> Dict[str, int]:
    labels = [int(graph.y.item()) for graph in graphs]
    return label_count_dict(labels, class_names)


def compute_class_weights(
    labels: Sequence[int],
    num_classes: int,
    power: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    counts = np.bincount(np.array([int(label) for label in labels], dtype=np.int64), minlength=num_classes)
    weights = np.ones(num_classes, dtype=np.float32)
    present = counts > 0
    if np.any(present):
        max_count = float(np.max(counts[present]))
        for idx, count in enumerate(counts):
            if count > 0:
                weights[idx] = float((max_count / float(count)) ** power)
            else:
                weights[idx] = 0.0
        mean_weight = float(np.mean(weights[present]))
        if mean_weight > 0.0:
            weights[present] = weights[present] / mean_weight
    summary = {
        "power": float(power),
        "counts": [int(item) for item in counts.tolist()],
        "weights": [float(item) for item in weights.tolist()],
    }
    return torch.tensor(weights, dtype=torch.float32), summary


def augment_training_records(
    records: Sequence[GraphRecord],
    train_indices: Sequence[int],
    class_names: Sequence[str],
    n_packets: int,
    include_sequence_features: bool,
    sequence_feature_size_clip: float,
    node_size_transform: str,
    node_size_clip: float,
    normalize_node_index: bool,
    packet_offset: int,
    augmentation_cfg: Dict[str, object],
    seed: int,
) -> Tuple[List[Data], List[int], Dict[str, object]]:
    targets = augmentation_cfg.get("targets", {})
    if not isinstance(targets, dict):
        targets = {}
    default_variants = int(augmentation_cfg.get("variants_per_sample", 0))
    seed_offset = int(augmentation_cfg.get("seed_offset", 0))
    max_augmented_per_class_value = augmentation_cfg.get("max_augmented_per_class")
    max_augmented_per_class = None if max_augmented_per_class_value in (None, "") else int(max_augmented_per_class_value)

    augmented_graphs: List[Data] = []
    augmented_labels: List[int] = []
    augmented_by_class = {class_name: 0 for class_name in class_names}
    skipped_by_class = {class_name: 0 for class_name in class_names}

    for source_position, record_idx in enumerate(train_indices):
        record = records[record_idx]
        variants = int(targets.get(record.class_name, default_variants))
        if variants <= 0:
            continue
        for variant_idx in range(variants):
            if max_augmented_per_class is not None and augmented_by_class[record.class_name] >= max_augmented_per_class:
                skipped_by_class[record.class_name] += 1
                continue
            rng = random.Random(seed + seed_offset + record_idx * 1009 + source_position * 131 + variant_idx * 17)
            augmentation_max_packets = n_packets + max(0, int(packet_offset)) if n_packets > 0 else 0
            augmented_line = augment_sequence_line(record.line, rng, augmentation_cfg, augmentation_max_packets)
            graph = graph_data_from_line(
                augmented_line,
                n_packets=n_packets,
                include_sequence_features=include_sequence_features,
                sequence_feature_size_clip=sequence_feature_size_clip,
                node_size_transform=node_size_transform,
                node_size_clip=node_size_clip,
                normalize_node_index=normalize_node_index,
                packet_offset=packet_offset,
            )
            if graph is None:
                skipped_by_class[record.class_name] += 1
                continue
            graph.y = torch.tensor([record.label], dtype=torch.long)
            augmented_graphs.append(graph)
            augmented_labels.append(record.label)
            augmented_by_class[record.class_name] += 1

    summary = {
        "enabled": bool(augmentation_cfg.get("enabled", False)),
        "targets": {str(key): int(value) for key, value in targets.items()},
        "variants_per_sample": default_variants,
        "size_jitter": float(augmentation_cfg.get("size_jitter", 0.0)),
        "dropout_prob": float(augmentation_cfg.get("dropout_prob", 0.0)),
        "duplicate_prob": float(augmentation_cfg.get("duplicate_prob", 0.0)),
        "crop_max_ratio": float(augmentation_cfg.get("crop_max_ratio", 0.0)),
        "min_tokens": int(augmentation_cfg.get("min_tokens", 8)),
        "max_augmented_per_class": max_augmented_per_class,
        "packet_offset": max(0, int(packet_offset)),
        "augmented_graphs": len(augmented_graphs),
        "augmented_by_class": augmented_by_class,
        "skipped_by_class": skipped_by_class,
    }
    return augmented_graphs, augmented_labels, summary


class FocalNLLLoss(torch.nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 2.0) -> None:
        super().__init__()
        if weight is None:
            self.register_buffer("weight", torch.empty(0))
        else:
            self.register_buffer("weight", weight.detach().clone())
        self.gamma = float(gamma)

    def forward(self, log_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = self.weight if self.weight.numel() else None
        nll = F.nll_loss(log_probs, target, weight=weight, reduction="none")
        pt = log_probs.gather(1, target.view(-1, 1)).squeeze(1).exp().clamp(min=1e-8, max=1.0)
        focal_base = (1.0 - pt).clamp(min=1e-8, max=1.0)
        return ((focal_base ** self.gamma) * nll).mean()


def build_training_criterion(
    loss_name: str,
    class_weights: Optional[torch.Tensor],
    focal_gamma: float,
    device: torch.device,
) -> torch.nn.Module:
    normalized = loss_name.lower()
    weights = class_weights.to(device) if class_weights is not None else None
    if normalized in ("nll", "plain_nll"):
        return torch.nn.NLLLoss()
    if normalized == "weighted_nll":
        return torch.nn.NLLLoss(weight=weights)
    if normalized in ("focal", "weighted_focal"):
        return FocalNLLLoss(weight=weights, gamma=focal_gamma)
    raise ValueError(f"Unsupported graph classifier loss: {loss_name}")


def evaluate_graph_loss_and_metrics(
    model: torch.nn.Module,
    graphs: Sequence[Data],
    criterion: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    num_classes: Optional[int],
) -> Tuple[float, Optional[Dict[str, float]]]:
    model.eval()
    total_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []
    loader = DataLoader(list(graphs), batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"))
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            total_loss += loss.item() * batch.num_graphs
            y_pred.extend(out.argmax(dim=1).cpu().numpy().tolist())
            y_true.extend(batch.y.cpu().numpy().tolist())
    val_loss = total_loss / max(len(graphs), 1)
    if num_classes is None:
        return val_loss, None
    labels = list(range(num_classes))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    return val_loss, {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "min_class_f1": float(np.min(f1)) if len(f1) else 0.0,
    }


def checkpoint_score(
    checkpoint_metric: str,
    train_loss: float,
    val_loss: Optional[float],
    val_metrics: Optional[Dict[str, float]],
    checkpoint_macro_weight: float = 0.7,
) -> Tuple[float, bool]:
    metric = checkpoint_metric.lower()
    if val_loss is None:
        return train_loss, False
    if metric == "val_loss":
        return val_loss, False
    if val_metrics is None:
        return train_loss, False
    if metric == "val_macro_f1":
        return val_metrics["macro_f1"], True
    if metric == "val_min_class_f1":
        return val_metrics["min_class_f1"], True
    if metric in ("val_macro_min_f1", "val_mixed_f1"):
        alpha = float(np.clip(checkpoint_macro_weight, 0.0, 1.0))
        return alpha * val_metrics["macro_f1"] + (1.0 - alpha) * val_metrics["min_class_f1"], True
    if metric == "train_loss":
        return train_loss, False
    raise ValueError(f"Unsupported checkpoint_metric: {checkpoint_metric}")


def train_graph_classifier(
    model: torch.nn.Module,
    train_graphs: Sequence[Data],
    val_graphs: Sequence[Data],
    batch_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
    logger: logging.Logger,
    ckpt_path: Union[str, os.PathLike[str]],
    loss_name: str = "nll",
    class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
    checkpoint_metric: str = "val_loss",
    checkpoint_macro_weight: float = 0.7,
    num_classes: Optional[int] = None,
    history_path: Optional[Union[str, os.PathLike[str]]] = None,
) -> torch.nn.Module:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = build_training_criterion(loss_name, class_weights, focal_gamma, device)
    best_state = None
    best_score: Optional[float] = None
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, object]] = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        loader = DataLoader(list(train_graphs), batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
        train_loss = train_loss / max(len(train_graphs), 1)

        if val_graphs:
            val_loss, val_metrics = evaluate_graph_loss_and_metrics(
                model,
                val_graphs,
                criterion,
                batch_size,
                device,
                num_classes,
            )
        else:
            val_loss = None
            val_metrics = None

        score, maximize = checkpoint_score(checkpoint_metric, train_loss, val_loss, val_metrics, checkpoint_macro_weight)
        improved = best_score is None or (score > best_score if maximize else score < best_score)

        if improved:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, ckpt_path)

        item: Dict[str, object] = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "checkpoint_metric": checkpoint_metric,
            "checkpoint_score": float(score),
            "is_best": bool(improved),
        }
        if val_loss is None:
            logger.info("epoch=%d train_loss=%.6f", epoch + 1, train_loss)
        else:
            item["val_loss"] = float(val_loss)
            if val_metrics is not None:
                item.update({f"val_{key}": value for key, value in val_metrics.items()})
                logger.info(
                    "epoch=%d train_loss=%.6f val_loss=%.6f val_macro_f1=%.6f val_min_class_f1=%.6f",
                    epoch + 1,
                    train_loss,
                    val_loss,
                    val_metrics["macro_f1"],
                    val_metrics["min_class_f1"],
                )
            else:
                logger.info("epoch=%d train_loss=%.6f val_loss=%.6f", epoch + 1, train_loss, val_loss)
        history.append(item)

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    if history_path is not None:
        dump_json(history_path, {
            "loss": loss_name,
            "checkpoint_metric": checkpoint_metric,
            "checkpoint_macro_weight": float(checkpoint_macro_weight),
            "best_score": float(best_score) if best_score is not None else None,
            "epochs": history,
        })
    model.load_state_dict(best_state)
    return model


def evaluate_graph_classifier(
    model: torch.nn.Module,
    test_graphs: Sequence[Data],
    class_names: Sequence[str],
    batch_size: int,
    device: torch.device,
) -> Dict[str, object]:
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

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
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
    return metrics


def write_report(
    output_dir: Union[str, os.PathLike[str]],
    class_names: Sequence[str],
    metrics: Dict[str, object],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(output_dir / "metrics.json", metrics)

    cm = np.array(metrics["confusion_matrix"], dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums * 100.0

    plt.figure(figsize=(max(8, len(class_names) * 0.75), max(6, len(class_names) * 0.6)))
    plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right", fontsize=8)
    plt.yticks(ticks, class_names, fontsize=8)
    threshold = cm_norm.max() / 2.0 if cm_norm.size else 0.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_norm[i, j]
            if val > 0:
                plt.text(
                    j,
                    i,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if val > threshold else "black",
                    fontsize=6,
                )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    lines = [
        f"Accuracy (overall): {metrics['accuracy'] * 100:.2f}%",
        f"Macro Precision: {metrics['macro_precision'] * 100:.2f}%",
        f"Macro Recall: {metrics['macro_recall'] * 100:.2f}%",
        f"Macro F1: {metrics['macro_f1'] * 100:.2f}%",
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
    (output_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def materialize_run_dir(config: Dict[str, object], config_hash: str) -> Path:
    runs_dir = ensure_dir(resolve_path(config["paths"]["runs_dir"]))  # type: ignore[index]
    experiment_name = str(config["experiment"]["name"])  # type: ignore[index]
    run_dir = runs_dir / f"{timestamp_tag()}_{experiment_name}_{config_hash[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
