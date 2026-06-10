"""Quick regression tests for Tab2 unified selection helpers."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import (  # noqa: E402
    _auto_choice_mask,
    _find_masks_at,
    _maintain_included_anti_chain,
    _make_auto_choice_state,
    _resolve_mask_pick,
)


def test_nested_masks():
    goose = np.zeros((100, 100), dtype=bool)
    goose[20:60, 20:60] = True
    both = goose.copy()
    both[55:70, 55:75] = True
    masks = [
        {"segmentation": both, "area": int(both.sum())},
        {"segmentation": goose, "area": int(goose.sum())},
    ]
    hits = _find_masks_at(masks, 30, 30)
    assert hits == [1, 0], hits

    idx, _ = _resolve_mask_pick(
        masks, 30, 30, _make_auto_choice_state([], []),
    )
    assert idx == 1, idx

    sel = _maintain_included_anti_chain(masks, [], idx)
    assert sel == [1], sel

    st = _make_auto_choice_state([], [], last_pick=(30, 30), cycle_idx=0)
    idx2, _ = _resolve_mask_pick(masks, 30, 30, st)
    assert idx2 == 0, idx2


def test_auto_choice_selected_only():
    goose = np.zeros((100, 100), dtype=bool)
    goose[20:60, 20:60] = True
    sock = np.zeros((100, 100), dtype=bool)
    sock[55:70, 55:75] = True
    both = goose | sock
    masks = [
        {"segmentation": both, "area": int(both.sum())},
        {"segmentation": goose, "area": int(goose.sum())},
    ]
    state = _make_auto_choice_state([1], [])
    mask = _auto_choice_mask(masks, state)
    assert mask[30, 30]
    assert not mask[60, 60]


if __name__ == "__main__":
    test_nested_masks()
    test_auto_choice_selected_only()
    print("all helper tests passed")
