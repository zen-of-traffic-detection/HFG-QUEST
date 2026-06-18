from __future__ import annotations

from typing import Dict, List


DATASET_VERSION = "squid_v2"

RAW_CLASSES: List[str] = [
    "benign",
    "deimos_download",
    "deimos_keylog",
    "deimos_recon",
    "deimos_screenshot",
    "deimos_shell",
    "deimos_upload",
    "merlin_download",
    "merlin_keylog",
    "merlin_recon",
    "merlin_screenshot",
    "merlin_shell",
    "merlin_upload",
    "sliver_download",
    "sliver_keylog",
    "sliver_recon",
    "sliver_screenshot",
    "sliver_shell",
    "sliver_upload",
]

BENIGN_CLASS = "benign"
MALICIOUS_CLASSES: List[str] = [name for name in RAW_CLASSES if name != BENIGN_CLASS]

STAGE1_CLASSES: List[str] = ["benign", "merlin", "deimos", "sliver"]
STAGE1_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(STAGE1_CLASSES)}

STAGE2_CLASSES: List[str] = MALICIOUS_CLASSES[:]
STAGE2_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(STAGE2_CLASSES)}


def raw_to_stage1(raw_class: str) -> str:
    if raw_class == BENIGN_CLASS:
        return BENIGN_CLASS
    return raw_class.split("_", 1)[0]


def raw_to_stage1_id(raw_class: str) -> int:
    return STAGE1_TO_ID[raw_to_stage1(raw_class)]


def raw_to_stage2_id(raw_class: str) -> int:
    return STAGE2_TO_ID[raw_class]


def stage1_display_names() -> List[str]:
    return STAGE1_CLASSES[:]


def stage2_display_names() -> List[str]:
    return STAGE2_CLASSES[:]


def dataset_summary() -> Dict[str, object]:
    return {
        "dataset_version": DATASET_VERSION,
        "raw_classes": RAW_CLASSES,
        "stage1_classes": STAGE1_CLASSES,
        "stage2_classes": STAGE2_CLASSES,
    }
