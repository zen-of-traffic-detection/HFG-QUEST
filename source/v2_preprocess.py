from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scapy.all import IP, IPv6, Packet, Raw, UDP, rdpcap

from v2_common import dump_json, ensure_dir, resolve_path, short_hash
from v2_dataset import DATASET_VERSION, RAW_CLASSES


class PacketRecord:
    def __init__(self, ts: float, src: str, sport: int, dst: str, dport: int, udp_plen: int) -> None:
        self.ts = ts
        self.src = src
        self.sport = sport
        self.dst = dst
        self.dport = dport
        self.udp_plen = udp_plen
        self.dir_cs = 0


def median(values: np.ndarray) -> float:
    return float(np.median(values)) if len(values) else float("nan")


def mad(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    center = median(values)
    return float(np.median(np.abs(values - center)))


def percentile(values: np.ndarray, q: float) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(values, q))


def safe_clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def natural_sort_key(path: Path) -> Tuple[int, Union[int, str]]:
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


def _parse_sample_limit(value: object) -> Optional[int]:
    return None if value in (None, "", "null") else int(value)  # type: ignore[arg-type]


def max_samples_from_config(config: Dict[str, object]) -> Optional[int]:
    max_samples = config["preprocess"].get("max_samples_per_class")  # type: ignore[index]
    return _parse_sample_limit(max_samples)


def max_samples_by_class_from_config(config: Dict[str, object]) -> Dict[str, Optional[int]]:
    preprocess_cfg = config["preprocess"]  # type: ignore[index]
    raw_policy = preprocess_cfg.get("max_samples_by_class", {})
    if raw_policy in (None, "", "null"):
        return {}
    if not isinstance(raw_policy, dict):
        raise TypeError("preprocess.max_samples_by_class must be a mapping")
    return {str(class_name): _parse_sample_limit(limit) for class_name, limit in raw_policy.items()}


def max_samples_for_class(config: Dict[str, object], class_name: str) -> Optional[int]:
    by_class = max_samples_by_class_from_config(config)
    if class_name in by_class:
        return by_class[class_name]
    return max_samples_from_config(config)


def cache_n_packets_from_config(config: Dict[str, object]) -> int:
    preprocess_cfg = config["preprocess"]  # type: ignore[index]
    return int(preprocess_cfg.get("cache_n_packets", preprocess_cfg["n_packets"]))


def preprocess_workers_from_config(config: Dict[str, object]) -> int:
    preprocess_cfg = config["preprocess"]  # type: ignore[index]
    value = preprocess_cfg.get("workers", preprocess_cfg.get("num_workers", 1))
    if value in (None, "", "null"):
        return 1
    requested = max(1, int(value))
    cpu_count = os.cpu_count() or requested
    return max(1, min(requested, cpu_count))


def awm_source_window_from_config(config: Dict[str, object]) -> Dict[str, int]:
    preprocess_cfg = config["preprocess"]  # type: ignore[index]
    offset = max(0, int(preprocess_cfg.get("stage2_awm_source_offset", 0)))
    n_packets = max(0, int(preprocess_cfg.get("stage2_awm_source_n_packets", 0)))
    return {"offset": offset, "n_packets": n_packets}


def slice_awm_source_tokens(tokens: List[str], source_window: Dict[str, int]) -> List[str]:
    offset = max(0, int(source_window.get("offset", 0)))
    n_packets = max(0, int(source_window.get("n_packets", 0)))
    sliced = list(tokens[offset:])
    if n_packets > 0:
        sliced = sliced[:n_packets]
    return sliced


def cache_only_enabled(config: Dict[str, object]) -> bool:
    preprocess_cfg = config.get("preprocess", {})
    return isinstance(preprocess_cfg, dict) and bool(preprocess_cfg.get("cache_only", False))


def cache_only_hb_key_from_config(config: Dict[str, object]) -> Optional[str]:
    preprocess_cfg = config.get("preprocess", {})
    if not isinstance(preprocess_cfg, dict):
        return None
    value = preprocess_cfg.get("cache_only_hb_key")
    return str(value) if value not in (None, "", "null") else None


def cache_only_awm_key_from_config(config: Dict[str, object]) -> Optional[str]:
    preprocess_cfg = config.get("preprocess", {})
    if not isinstance(preprocess_cfg, dict):
        return None
    value = preprocess_cfg.get("cache_only_awm_key")
    return str(value) if value not in (None, "", "null") else None


def cache_inventory_path_from_config(config: Dict[str, object]) -> Path:
    paths_cfg = config.get("paths", {})
    if isinstance(paths_cfg, dict) and paths_cfg.get("cache_inventory"):
        return resolve_path(paths_cfg["cache_inventory"])  # type: ignore[arg-type]
    return resolve_path("config/cache_inventory.json")


def cache_inventory_line_count(path_value: object) -> int:
    if path_value in (None, "", "null"):
        return 0
    path = resolve_path(str(path_value))
    try:
        with path.open("rb") as handle:
            return sum(1 for _ in handle)
    except FileNotFoundError as exc:
        raise RuntimeError(f"cache inventory references a missing cache file: {path}") from exc


def cache_inventory_from_config(config: Dict[str, object]) -> Dict[str, object]:
    path = cache_inventory_path_from_config(config)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"cache_only=True but cache inventory is missing: {path}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"cache inventory must be a JSON object: {path}")
    return data


def list_class_pcaps(data_dir: Path, class_name: str, max_samples: Optional[int]) -> List[Path]:
    class_dir = data_dir / class_name
    if not class_dir.exists():
        return []
    files = sorted(
        [path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in {".pcap", ".pcapng"}],
        key=natural_sort_key,
    )
    if max_samples is not None:
        return files[:max_samples]
    return files


def raw_data_inventory(config: Dict[str, object], include_files: bool = False) -> Dict[str, object]:
    if cache_only_enabled(config):
        cache_inventory = cache_inventory_from_config(config)
        classes: Dict[str, object] = {}
        hash_classes: Dict[str, object] = {}
        total_files = 0
        total_size_bytes = 0
        for class_name in RAW_CLASSES:
            item = dict(cache_inventory.get("classes", {}).get(class_name, {}))  # type: ignore[union-attr]
            if "samples" in item:
                samples = int(item.get("samples", 0))
            else:
                samples = cache_inventory_line_count(item.get("before_file", f"{class_name}.txt"))
            before_size = int(item.get("before_size_bytes", 0))
            after_size = int(item.get("after_size_bytes", 0))
            file_items = [
                {"name": item.get("before_file", f"{class_name}.txt"), "size_bytes": before_size},
                {"name": item.get("after_file", f"{class_name}_packets.txt"), "size_bytes": after_size},
            ]
            classes[class_name] = {
                "input_files": samples,
                "total_size_bytes": before_size + after_size,
                "class_inventory_hash": short_hash(file_items),
                "max_samples": None,
                "cache_only": True,
            }
            if include_files:
                classes[class_name]["files"] = file_items  # type: ignore[index]
            hash_classes[class_name] = file_items
            total_files += samples
            total_size_bytes += before_size + after_size
        raw_inventory_hash = str(cache_inventory.get("cache_inventory_hash") or short_hash(hash_classes))
        return {
            "dataset_version": DATASET_VERSION,
            "config_dataset_version": config["experiment"]["dataset_version"],  # type: ignore[index]
            "source_data_dir": "cache_only",
            "max_samples_per_class": None,
            "max_samples_by_class": {},
            "raw_inventory_hash": raw_inventory_hash,
            "total_files": total_files,
            "total_size_bytes": total_size_bytes,
            "classes": classes,
            "cache_only": True,
            "hb_key": cache_inventory.get("hb_key"),
            "awm_key": cache_inventory.get("awm_key"),
        }

    data_dir = resolve_path(config["paths"]["data_dir"])  # type: ignore[index]
    max_samples = max_samples_from_config(config)
    max_samples_by_class = max_samples_by_class_from_config(config)

    classes: Dict[str, object] = {}
    hash_classes: Dict[str, object] = {}
    total_files = 0
    total_size_bytes = 0

    for class_name in RAW_CLASSES:
        class_max_samples = max_samples_by_class.get(class_name, max_samples)
        files = list_class_pcaps(data_dir, class_name, class_max_samples)
        file_items = [{"name": path.name, "size_bytes": path.stat().st_size} for path in files]
        class_size = sum(item["size_bytes"] for item in file_items)
        class_summary: Dict[str, object] = {
            "input_files": len(file_items),
            "total_size_bytes": class_size,
            "class_inventory_hash": short_hash(file_items),
            "max_samples": class_max_samples,
        }
        if include_files:
            class_summary["files"] = file_items
        classes[class_name] = class_summary
        hash_classes[class_name] = file_items
        total_files += len(file_items)
        total_size_bytes += class_size

    hash_payload = {
        "dataset_version": config["experiment"]["dataset_version"],  # type: ignore[index]
        "source_data_dir": str(data_dir),
        "max_samples_per_class": max_samples,
        "classes": hash_classes,
    }
    if max_samples_by_class:
        hash_payload["max_samples_by_class"] = max_samples_by_class
    raw_inventory_hash = short_hash(hash_payload)

    return {
        "dataset_version": DATASET_VERSION,
        "config_dataset_version": config["experiment"]["dataset_version"],  # type: ignore[index]
        "source_data_dir": str(data_dir),
        "max_samples_per_class": max_samples,
        "max_samples_by_class": max_samples_by_class,
        "raw_inventory_hash": raw_inventory_hash,
        "total_files": total_files,
        "total_size_bytes": total_size_bytes,
        "classes": classes,
    }


def get_udp_payload_len(pkt: Packet) -> int:
    try:
        udp_len = int(pkt[UDP].len) if int(pkt[UDP].len) > 0 else None
    except Exception:
        udp_len = None
    if udp_len is not None:
        return max(0, udp_len - 8)
    if Raw in pkt and isinstance(pkt[Raw].load, (bytes, bytearray)):
        return len(pkt[Raw].load)
    return 0


def read_flow_from_pcap(pcap_path: str) -> List[PacketRecord]:
    packets = rdpcap(pcap_path)
    flow: List[PacketRecord] = []
    for pkt in packets:
        try:
            if UDP not in pkt:
                continue
            ts = float(pkt.time)
            if IP in pkt:
                src = pkt[IP].src
                dst = pkt[IP].dst
            elif IPv6 in pkt:
                src = pkt[IPv6].src
                dst = pkt[IPv6].dst
            else:
                continue
            sport = int(pkt[UDP].sport)
            dport = int(pkt[UDP].dport)
            flow.append(PacketRecord(ts, src, sport, dst, dport, get_udp_payload_len(pkt)))
        except Exception:
            continue
    flow.sort(key=lambda item: item.ts)
    return flow


def encode_before_cache_line(
    pcap_path: str,
    n_packets: int,
    heartbeat_cfg: Dict[str, object],
) -> Optional[str]:
    flow = read_flow_from_pcap(pcap_path)
    if n_packets > 0:
        flow = flow[:n_packets]
    if not flow:
        return None
    if not bool(heartbeat_cfg.get("enabled", True)):
        return encode_flow_tokens(flow, [0] * len(flow))
    alpha = float(heartbeat_cfg["weights"]["alpha"])  # type: ignore[index]
    beta = float(heartbeat_cfg["weights"]["beta"])  # type: ignore[index]
    gamma = float(heartbeat_cfg["weights"]["gamma"])  # type: ignore[index]
    hb_percentile = float(heartbeat_cfg["percentile_threshold"])
    components = precompute_components(flow, heartbeat_cfg)
    labels = labels_from_components(components, len(flow), alpha, beta, gamma, hb_percentile)
    return encode_flow_tokens(flow, labels)


def encode_before_cache_worker(args: Tuple[int, str, int, Dict[str, object]]) -> Tuple[int, Optional[str], Optional[str]]:
    index, pcap_path, n_packets, heartbeat_cfg = args
    try:
        return index, encode_before_cache_line(pcap_path, n_packets, heartbeat_cfg), None
    except Exception as exc:
        return index, None, f"{Path(pcap_path).name}: {type(exc).__name__}: {exc}"


def infer_client_direction(flow: List[PacketRecord], rhythm_cfg: Dict[str, float], size_cfg: Dict[str, float]) -> Tuple[str, int]:
    large = [pkt for pkt in flow if pkt.udp_plen >= 1200]
    if large:
        first = min(large, key=lambda pkt: pkt.ts)
        return first.src, first.sport

    if not flow:
        return ("", 0)

    first_pkt = flow[0]
    candidate_a = (first_pkt.src, first_pkt.sport)

    counts: Dict[Tuple[str, int], int] = {}
    for pkt in flow:
        for endpoint in ((pkt.src, pkt.sport), (pkt.dst, pkt.dport)):
            if endpoint != candidate_a:
                counts[endpoint] = counts.get(endpoint, 0) + 1
    candidate_b = max(counts.items(), key=lambda item: item[1])[0] if counts else candidate_a

    def rhythm_tightness(client_endpoint: Tuple[str, int]) -> float:
        c2s = [pkt for pkt in flow if (pkt.src, pkt.sport) == client_endpoint]
        if len(c2s) < 5:
            return float("inf")
        lens = np.array([pkt.udp_plen for pkt in c2s], dtype=float)
        p30 = percentile(lens, size_cfg["small_candidate_percentile"])
        small = [pkt for pkt in c2s if pkt.udp_plen <= p30]
        if len(small) < 5:
            k = max(5, len(c2s) // 8 or 1)
            small = sorted(c2s, key=lambda pkt: pkt.udp_plen)[:k]
        if len(small) < 4:
            return float("inf")
        stamps = np.array([pkt.ts for pkt in small], dtype=float)
        best_dispersion = float("inf")
        max_order = int(rhythm_cfg["max_order"])
        for order in range(1, min(max_order, len(stamps) - 1) + 1):
            delta = np.diff(stamps, n=order)
            if len(delta) == 0:
                continue
            best_dispersion = min(best_dispersion, mad(delta))
        return best_dispersion

    return candidate_a if rhythm_tightness(candidate_a) <= rhythm_tightness(candidate_b) else candidate_b


def precompute_components(
    flow: List[PacketRecord],
    heartbeat_cfg: Dict[str, object],
) -> Dict[str, Dict[int, float]]:
    weights = heartbeat_cfg["weights"]  # type: ignore[index]
    size_cfg = heartbeat_cfg["size"]  # type: ignore[index]
    rhythm_cfg = heartbeat_cfg["rhythm"]  # type: ignore[index]
    direction_cfg = heartbeat_cfg["direction"]  # type: ignore[index]

    client_ip, client_port = infer_client_direction(flow, rhythm_cfg, size_cfg)
    for pkt in flow:
        pkt.dir_cs = +1 if (pkt.src == client_ip and pkt.sport == client_port) else -1

    c2s = [pkt for pkt in flow if pkt.dir_cs == +1]
    s2c = [pkt for pkt in flow if pkt.dir_cs == -1]

    def size_scores(sub_flow: List[PacketRecord]) -> Dict[int, float]:
        if not sub_flow:
            return {}
        lens = np.array([pkt.udp_plen for pkt in sub_flow], dtype=float)
        q25 = percentile(lens, size_cfg["q25"])
        q75 = percentile(lens, size_cfg["q75"])
        denom = (q75 - q25) if q75 > q25 else max(1.0, q75)
        scores: Dict[int, float] = {}
        for idx, pkt in enumerate(sub_flow):
            z = (pkt.udp_plen - q25) / denom
            scores[idx] = safe_clip(1.0 - z, 0.0, 1.0)
        return scores

    size_c2s = size_scores(c2s)
    size_s2c = size_scores(s2c)
    idx_c2s = {id(pkt): idx for idx, pkt in enumerate(c2s)}
    idx_s2c = {id(pkt): idx for idx, pkt in enumerate(s2c)}
    size_score: Dict[int, float] = {}
    for idx, pkt in enumerate(flow):
        if pkt.dir_cs == +1:
            size_score[idx] = size_c2s.get(idx_c2s[id(pkt)], 0.0)
        else:
            size_score[idx] = size_s2c.get(idx_s2c[id(pkt)], 0.0)

    def small_candidates(sub_flow: List[PacketRecord]) -> List[PacketRecord]:
        if not sub_flow:
            return []
        lens = np.array([pkt.udp_plen for pkt in sub_flow], dtype=float)
        p_small = percentile(lens, size_cfg["small_candidate_percentile"])
        candidates = [pkt for pkt in sub_flow if pkt.udp_plen <= p_small]
        if len(candidates) < 5:
            k = max(5, len(sub_flow) // 8 or 1)
            candidates = sorted(sub_flow, key=lambda pkt: pkt.udp_plen)[:k]
        lens_small = np.array([pkt.udp_plen for pkt in candidates], dtype=float)
        threshold = median(lens_small) + size_cfg["mad_multiplier"] * mad(lens_small)
        return [pkt for pkt in candidates if pkt.udp_plen <= threshold]

    candidates = small_candidates(c2s)
    rhythm_score: Dict[int, float] = {idx: 0.0 for idx in range(len(flow))}
    best_period = None
    anchor = None

    if len(candidates) >= 4:
        timestamps = np.array([pkt.ts for pkt in sorted(candidates, key=lambda item: item.ts)], dtype=float)
        best_dispersion = float("inf")
        max_order = int(rhythm_cfg["max_order"])
        for order in range(1, min(max_order, len(timestamps) - 1) + 1):
            delta = np.diff(timestamps, n=order)
            if len(delta) == 0:
                continue
            dispersion = mad(delta)
            center = median(delta)
            if center > 0 and dispersion < best_dispersion:
                best_dispersion = dispersion
                best_period = center
        if best_period is not None and rhythm_cfg["min_period"] <= best_period <= rhythm_cfg["max_period"]:
            anchor = float(min(candidates, key=lambda pkt: pkt.ts).ts)

    if best_period is not None and anchor is not None:
        def phase_residual(ts: float) -> float:
            modulo = (ts - anchor) % best_period
            return min(modulo, best_period - modulo)

        residuals = np.array([phase_residual(pkt.ts) for pkt in candidates], dtype=float)
        eps = max(
            rhythm_cfg["epsilon_ratio"] * best_period,
            rhythm_cfg["epsilon_mad_multiplier"] * mad(residuals),
        )
        for idx, pkt in enumerate(flow):
            residual = phase_residual(pkt.ts)
            value = 1.0 - (residual / eps)
            rhythm_score[idx] = safe_clip(value, 0.0, 1.0) if residual <= eps else 0.0

    dir_score: Dict[int, float] = {}
    for idx, pkt in enumerate(flow):
        dir_score[idx] = direction_cfg["c2s_score"] if pkt.dir_cs == +1 else direction_cfg["s2c_score"]

    return {
        "size_score": size_score,
        "rhythm_score": rhythm_score,
        "dir_score": dir_score,
        "weights": {
            "alpha": weights["alpha"],
            "beta": weights["beta"],
            "gamma": weights["gamma"],
        },
    }


def labels_from_components(
    components: Dict[str, Dict[int, float]],
    n: int,
    alpha: float,
    beta: float,
    gamma: float,
    hb_percentile: float,
) -> List[int]:
    hb_score: Dict[int, float] = {}
    for idx in range(n):
        hb_score[idx] = (
            alpha * components["size_score"].get(idx, 0.0)
            + beta * components["rhythm_score"].get(idx, 0.0)
            + gamma * components["dir_score"].get(idx, 0.0)
        )
    arr = np.array(list(hb_score.values()), dtype=float)
    threshold = percentile(arr, hb_percentile) if len(arr) else 1.0
    return [int(hb_score[idx] >= threshold) for idx in range(n)]


def encode_flow_tokens(flow: List[PacketRecord], hb_labels: List[int]) -> str:
    tokens: List[str] = []
    for pkt, hb in zip(flow, hb_labels):
        size = pkt.udp_plen if pkt.udp_plen >= 0 else 0
        if hb == 1:
            tokens.append(f"+{size}" if pkt.dir_cs == +1 else f"+-{size}")
        else:
            tokens.append(f"{size}" if pkt.dir_cs == +1 else f"-{size}")
    return " ".join(tokens)


def parse_token_direction_and_size(token: str) -> Tuple[int, int]:
    direction = +1
    stripped = token.strip()
    if stripped.startswith("+-") or stripped.startswith("-"):
        direction = -1
    core = stripped.lstrip("+-")
    try:
        value = int(core)
    except ValueError:
        value = 0
    return direction, abs(value)


def sliding_windows(n_packets: int, win_len: int, overlap: float) -> List[Tuple[int, int]]:
    if n_packets <= 0 or win_len <= 0:
        return []
    stride = max(1, int(math.floor(win_len * (1.0 - overlap))))
    ranges: List[Tuple[int, int]] = []
    start = 0
    while start + win_len <= n_packets:
        end = start + win_len - 1
        ranges.append((start, end))
        start += stride
    if not ranges or ranges[-1][1] != n_packets - 1:
        end = n_packets - 1
        start = max(0, end - win_len + 1)
        if not ranges or ranges[-1] != (start, end):
            ranges.append((start, end))
    return sorted(set(ranges))


def feature_window(tokens: List[str], start: int, end: int) -> Tuple[float, float, float, float]:
    n = end - start + 1
    if n <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    bins = [0] * 16
    sizes: List[int] = []
    prev_dir = None
    direction_changes = 0
    for idx in range(start, end + 1):
        direction, length = parse_token_direction_and_size(tokens[idx])
        sizes.append(length)
        bins[length % 16] += 1
        if prev_dir is not None and prev_dir != direction:
            direction_changes += 1
        prev_dir = direction
    probabilities = np.array(bins, dtype=float) / float(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.nansum(np.where(probabilities > 0, probabilities * np.log2(probabilities), 0.0))
    return (float(n), float(direction_changes), float(np.var(sizes)) if n > 1 else 0.0, float(entropy))


def zscore_per_feature(features: np.ndarray) -> np.ndarray:
    mu = np.nanmean(features, axis=0)
    sigma = np.nanstd(features, axis=0)
    sigma[sigma == 0] = 1.0
    return np.where(np.isfinite((features - mu) / sigma), (features - mu) / sigma, 0.0)


def uniform_pick_low_indices(low_indices: List[int], need: int) -> List[int]:
    if need <= 0 or not low_indices:
        return []
    if need >= len(low_indices):
        return low_indices
    step = len(low_indices) / float(need)
    picks: List[int] = []
    position = step / 2.0
    for _ in range(need):
        picks.append(low_indices[int(position)])
        position += step
    return sorted(set(picks))


def run_awm_kws_on_flow(tokens: List[str], awm_cfg: Dict[str, object]) -> List[int]:
    if not tokens:
        return []
    if not bool(awm_cfg.get("enabled", True)):
        return list(range(len(tokens)))
    win_len = int(awm_cfg["win_len"])
    overlap = float(awm_cfg["overlap"])
    weights = np.array(awm_cfg["weights"], dtype=float).reshape(4)
    l_min = int(awm_cfg["L_min"])
    tail_k = int(awm_cfg["tail_k"])
    score_percentile = float(awm_cfg.get("score_percentile", 80.0))
    window_ranges = sliding_windows(len(tokens), win_len, overlap)
    features = np.zeros((len(window_ranges), 4), dtype=float)
    for idx, (start, end) in enumerate(window_ranges):
        features[idx, :] = feature_window(tokens, start, end)
    scores = (zscore_per_feature(features) * weights).sum(axis=1)
    threshold = percentile(scores, score_percentile)
    high = [idx for idx, value in enumerate(scores) if value >= threshold]
    low = [idx for idx, value in enumerate(scores) if value < threshold]
    selected = set(high)
    for idx in high:
        if idx - 1 >= 0:
            selected.add(idx - 1)
        if idx + 1 < len(window_ranges):
            selected.add(idx + 1)
    for idx in range(max(0, len(window_ranges) - tail_k), len(window_ranges)):
        selected.add(idx)
    if len(selected) < l_min and low:
        need = l_min - len(selected)
        candidates = [idx for idx in low if idx not in selected]
        for idx in uniform_pick_low_indices(candidates, need):
            selected.add(idx)
    ordered_ranges = [window_ranges[idx] for idx in sorted(selected)]
    packet_indices: List[int] = []
    seen = set()
    for start, end in ordered_ranges:
        for idx in range(start, end + 1):
            if idx not in seen:
                seen.add(idx)
                packet_indices.append(idx)
    return packet_indices


def hb_key_from_config(config: Dict[str, object]) -> str:
    if cache_only_enabled(config):
        hb_key = cache_only_hb_key_from_config(config)
        if hb_key:
            return hb_key
    inventory = raw_data_inventory(config)
    cache_n_packets = cache_n_packets_from_config(config)
    payload = {
        "dataset_version": config["experiment"]["dataset_version"],  # type: ignore[index]
        "cache_n_packets": cache_n_packets,
        "max_samples_per_class": config["preprocess"].get("max_samples_per_class"),  # type: ignore[index]
        "max_samples_by_class": config["preprocess"].get("max_samples_by_class"),  # type: ignore[index]
        "raw_inventory_hash": inventory["raw_inventory_hash"],
        "heartbeat": config["heartbeat"],
    }
    return short_hash(payload)


def awm_key_from_config(config: Dict[str, object], hb_key: str) -> str:
    if cache_only_enabled(config):
        awm_key = cache_only_awm_key_from_config(config)
        if awm_key:
            cache_dir = after_cache_dir(config, awm_key)
            manifest_path = cache_dir.parent / "manifest.json"
            manifest = _load_manifest(manifest_path)
            source_window = awm_source_window_from_config(config)
            if (
                all(path.exists() for path in _expected_after_files(cache_dir))
                and manifest is not None
                and manifest.get("hb_key") == hb_key
                and manifest.get("stage2_awm_source") == source_window
                and manifest.get("awm_kws") == config["awm_kws"]
            ):
                return awm_key
    payload = {
        "hb_key": hb_key,
        "awm_kws": config["awm_kws"],
        "stage2_awm_source": awm_source_window_from_config(config),
    }
    return short_hash(payload)


def before_cache_dir(config: Dict[str, object], hb_key: str) -> Path:
    root = resolve_path(config["paths"]["preprocess_dir"])  # type: ignore[index]
    return root / "hb_cache" / hb_key / "before_awm_kws"


def after_cache_dir(config: Dict[str, object], awm_key: str) -> Path:
    root = resolve_path(config["paths"]["preprocess_dir"])  # type: ignore[index]
    return root / "awm_cache" / awm_key / "after_awm_kws"


def _expected_before_files(directory: Path) -> List[Path]:
    return [directory / f"{class_name}.txt" for class_name in RAW_CLASSES]


def _expected_after_files(directory: Path) -> List[Path]:
    return [directory / f"{class_name}_packets.txt" for class_name in RAW_CLASSES]


def _load_manifest(path: Path) -> Optional[Dict[str, object]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_before_class_cache(
    files: List[Path],
    out_path: Path,
    n_packets: int,
    heartbeat_cfg: Dict[str, object],
    workers: int,
) -> Tuple[int, int]:
    if not files:
        out_path.write_text("", encoding="utf-8")
        return 0, 0

    worker_count = min(max(1, int(workers)), len(files))
    results: List[Tuple[Optional[str], Optional[str]]] = [(None, None) for _ in files]

    if worker_count == 1:
        for index, file_path in enumerate(files):
            _, line, error = encode_before_cache_worker((index, str(file_path), n_packets, heartbeat_cfg))
            results[index] = (line, error)
    else:
        try:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(encode_before_cache_worker, (index, str(file_path), n_packets, heartbeat_cfg))
                    for index, file_path in enumerate(files)
                ]
                for future in as_completed(futures):
                    index, line, error = future.result()
                    results[index] = (line, error)
        except BrokenProcessPool:
            results = [(None, None) for _ in files]
            for index, file_path in enumerate(files):
                _, line, error = encode_before_cache_worker((index, str(file_path), n_packets, heartbeat_cfg))
                results[index] = (line, error)

    written = 0
    errors = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for line, error in results:
            if error is not None:
                errors += 1
                continue
            if line is None:
                continue
            handle.write(line + "\n")
            written += 1
    return written, errors


def ensure_before_cache(config: Dict[str, object], logger) -> Tuple[str, Path]:
    if cache_only_enabled(config):
        hb_key = hb_key_from_config(config)
        cache_dir = before_cache_dir(config, hb_key)
        if all(path.exists() for path in _expected_before_files(cache_dir)):
            logger.info("Reusing cache-only heartbeat cache: %s", cache_dir)
            return hb_key, cache_dir
        raise RuntimeError(f"cache_only=True but heartbeat cache files are missing: {cache_dir}")
    inventory = raw_data_inventory(config)
    hb_key = hb_key_from_config(config)
    cache_dir = before_cache_dir(config, hb_key)
    manifest_path = cache_dir.parent / "manifest.json"
    manifest = _load_manifest(manifest_path)
    if (
        all(path.exists() for path in _expected_before_files(cache_dir))
        and manifest is not None
        and manifest.get("raw_inventory_hash") == inventory["raw_inventory_hash"]
    ):
        logger.info("Reusing heartbeat cache: %s", cache_dir)
        return hb_key, cache_dir
    if manifest is not None and manifest.get("raw_inventory_hash") != inventory["raw_inventory_hash"]:
        logger.info("Heartbeat cache inventory mismatch; rebuilding: %s", cache_dir)

    ensure_dir(cache_dir)
    n_packets = cache_n_packets_from_config(config)
    workers = preprocess_workers_from_config(config)

    heartbeat_cfg = config["heartbeat"]  # type: ignore[index]
    data_dir = resolve_path(config["paths"]["data_dir"])  # type: ignore[index]
    logger.info("Building heartbeat cache with workers=%d", workers)

    manifest = {
        "dataset_version": DATASET_VERSION,
        "hb_key": hb_key,
        "source_data_dir": str(data_dir),
        "cache_n_packets": n_packets,
        "preprocess_workers": workers,
        "raw_inventory_hash": inventory["raw_inventory_hash"],
        "raw_inventory": inventory,
        "heartbeat": heartbeat_cfg,
        "beacon_extraction_disabled": not bool(heartbeat_cfg.get("enabled", True)),
        "classes": {},
    }

    for class_name in RAW_CLASSES:
        class_max_samples = max_samples_for_class(config, class_name)
        files = list_class_pcaps(data_dir, class_name, class_max_samples)

        out_path = cache_dir / f"{class_name}.txt"
        written, errors = write_before_class_cache(files, out_path, n_packets, heartbeat_cfg, workers)
        manifest["classes"][class_name] = {
            "input_files": len(files),
            "written_lines": written,
            "errors": errors,
            "max_samples": class_max_samples,
        }
        logger.info("before_awm_kws %s: %d/%d errors=%d workers=%d", class_name, written, len(files), errors, min(workers, max(len(files), 1)))

    return hb_key, cache_dir


def ensure_after_cache(config: Dict[str, object], logger, hb_key: Optional[str] = None) -> Tuple[str, Path]:
    if hb_key is None:
        hb_key, _ = ensure_before_cache(config, logger)
    inventory = raw_data_inventory(config)
    awm_key = awm_key_from_config(config, hb_key)
    cache_dir = after_cache_dir(config, awm_key)
    manifest_path = cache_dir.parent / "manifest.json"
    manifest = _load_manifest(manifest_path)
    source_window = awm_source_window_from_config(config)
    if (
        cache_only_enabled(config)
        and all(path.exists() for path in _expected_after_files(cache_dir))
        and manifest is not None
        and manifest.get("hb_key") == hb_key
        and manifest.get("stage2_awm_source") == source_window
        and manifest.get("awm_kws") == config["awm_kws"]
    ):
        logger.info("Reusing cache-only AWM cache: %s", cache_dir)
        return awm_key, cache_dir
    if (
        all(path.exists() for path in _expected_after_files(cache_dir))
        and manifest is not None
        and manifest.get("hb_key") == hb_key
        and manifest.get("raw_inventory_hash") == inventory["raw_inventory_hash"]
        and manifest.get("stage2_awm_source") == source_window
    ):
        logger.info("Reusing AWM cache: %s", cache_dir)
        return awm_key, cache_dir
    if manifest is not None and manifest.get("raw_inventory_hash") != inventory["raw_inventory_hash"]:
        logger.info("AWM cache inventory mismatch; rebuilding: %s", cache_dir)

    ensure_dir(cache_dir)
    _, before_dir = ensure_before_cache(config, logger)
    awm_cfg = config["awm_kws"]  # type: ignore[index]
    logger.info(
        "Building AWM cache source_offset=%d source_n_packets=%d",
        source_window["offset"],
        source_window["n_packets"],
    )
    manifest = {
        "dataset_version": DATASET_VERSION,
        "hb_key": hb_key,
        "awm_key": awm_key,
        "raw_inventory_hash": inventory["raw_inventory_hash"],
        "raw_inventory": inventory,
        "awm_kws": awm_cfg,
        "stage2_awm_source": source_window,
        "classes": {},
    }

    for class_name in RAW_CLASSES:
        src_path = before_dir / f"{class_name}.txt"
        dst_path = cache_dir / f"{class_name}_packets.txt"
        written = 0
        skipped = 0
        source_lengths: List[int] = []
        selected_lengths: List[int] = []
        with src_path.open("r", encoding="utf-8") as src, dst_path.open("w", encoding="utf-8") as dst:
            for line in src:
                tokens = [token for token in line.strip().split() if token]
                if not tokens:
                    skipped += 1
                    continue
                source_tokens = slice_awm_source_tokens(tokens, source_window)
                if not source_tokens:
                    skipped += 1
                    continue
                picked = run_awm_kws_on_flow(source_tokens, awm_cfg)
                dst.write(" ".join(source_tokens[idx] for idx in picked) + "\n")
                written += 1
                source_lengths.append(len(source_tokens))
                selected_lengths.append(len(picked))

        class_manifest: Dict[str, object] = {"written_lines": written, "skipped_lines": skipped}
        if source_lengths:
            class_manifest.update(
                {
                    "source_tokens_min": min(source_lengths),
                    "source_tokens_max": max(source_lengths),
                    "source_tokens_mean": float(sum(source_lengths) / len(source_lengths)),
                    "selected_tokens_min": min(selected_lengths),
                    "selected_tokens_max": max(selected_lengths),
                    "selected_tokens_mean": float(sum(selected_lengths) / len(selected_lengths)),
                }
            )
        manifest["classes"][class_name] = class_manifest
        logger.info(
            "after_awm_kws %s: %d skipped=%d source_mean=%.2f selected_mean=%.2f",
            class_name,
            written,
            skipped,
            float(sum(source_lengths) / len(source_lengths)) if source_lengths else 0.0,
            float(sum(selected_lengths) / len(selected_lengths)) if selected_lengths else 0.0,
        )

    dump_json(manifest_path, manifest)
    return awm_key, cache_dir
