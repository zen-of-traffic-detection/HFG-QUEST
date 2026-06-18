from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from v2_common import (
    configure_logger,
    dump_json,
    load_yaml_or_json,
    materialize_run_dir,
    resolve_path,
    set_seed,
    short_hash,
)
from v2_dataset import dataset_summary
from v2_preprocess import ensure_after_cache, ensure_before_cache, raw_data_inventory
from v2_stage1 import run_stage1
from v2_stage2 import run_stage2


def run_experiment(config: Dict[str, object], config_source: str) -> Path:
    config_hash = short_hash(config)
    run_dir = materialize_run_dir(config, config_hash)
    logger = configure_logger(run_dir / "run.log")
    logger.info("Starting v2 experiment run in %s", run_dir)
    logger.info("Config source: %s", config_source)
    logger.info("Config hash: %s", config_hash)

    dump_json(run_dir / "config_resolved.json", config)
    dump_json(run_dir / "dataset_summary.json", dataset_summary())
    raw_inventory = raw_data_inventory(config)

    training_cfg = config["training"]  # type: ignore[index]
    set_seed(int(training_cfg["seed"]))

    hb_key, before_dir = ensure_before_cache(config, logger)
    awm_key, after_dir = ensure_after_cache(config, logger, hb_key=hb_key)

    stage_metrics = {}
    stage1_metrics, stage1_dir = run_stage1(config, before_dir, run_dir, logger)
    stage_metrics["stage1"] = {
        "metrics": stage1_metrics,
        "dir": str(stage1_dir),
    }

    stage2_metrics, stage2_dir = run_stage2(config, after_dir, run_dir, logger)
    stage_metrics["stage2"] = {"metrics": stage2_metrics, "dir": str(stage2_dir)}

    summary = {
        "hb_key": hb_key,
        "awm_key": awm_key,
        "raw_inventory_hash": raw_inventory.get("raw_inventory_hash"),
        "before_cache_dir": str(before_dir),
        "after_cache_dir": str(after_dir),
        "run_dir": str(run_dir),
        "stages": stage_metrics,
    }
    robustness_cfg = config.get("robustness", {})
    if isinstance(robustness_cfg, dict) and bool(robustness_cfg.get("enabled", False)):
        summary["robustness"] = {
            "enabled": True,
            "ratio": robustness_cfg.get("ratio"),
            "seed": robustness_cfg.get("seed", training_cfg.get("seed")),
            "variants": robustness_cfg.get("variants", []),
        }
        for stage_name in ("stage1", "stage2"):
            perturbation_path = run_dir / stage_name / "perturbation_summary.json"
            if perturbation_path.exists():
                summary["robustness"][stage_name] = json.loads(perturbation_path.read_text(encoding="utf-8"))  # type: ignore[index]
    dump_json(run_dir / "run_summary.json", summary)
    logger.info("Finished run. Summary saved to %s", run_dir / "run_summary.json")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HFG-QUEST SQUID v2 experiment")
    parser.add_argument("--config", required=True, help="Path to a YAML/JSON config file.")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    config = load_yaml_or_json(config_path)
    run_dir = run_experiment(config, str(config_path))
    print(run_dir)


if __name__ == "__main__":
    main()
