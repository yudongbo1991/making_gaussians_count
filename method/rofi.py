"""TRAIN-response ranked occupancy fusion; preserve every SfM cell."""

import numpy as np
from method.io import keys
from method.constants import KERNEL_ALPHA_CUTOFF


def ranked_pool(sfm, xyz, opacity, relevance, positive_views, width):
    valid = (positive_views >= 2) & (relevance[:, 0] > 0) & np.isfinite(xyz).all(1)
    ids = np.flatnonzero(valid)
    cells, inverse = np.unique(
        np.rint(xyz[valid] / np.float32(width)).astype(np.int64), axis=0, return_inverse=True
    )
    summed = np.zeros((len(cells), 3))
    np.add.at(summed, inverse, relevance[valid])
    order = np.lexsort((-relevance[valid, 0], inverse))
    first = np.r_[True, np.diff(inverse[order]) != 0] if len(order) else np.zeros(0, bool)
    representative = ids[order[first]]
    assert len(representative) == len(cells)
    novel = ~np.isin(keys(cells), keys(sfm))
    overlap = int((~novel).sum())
    cells, summed, representative = (cells[novel], summed[novel], representative[novel])
    utility = 0.5 * summed[:, 0] / max(summed[:, 0].sum(), 1e-30) + 0.5 * summed[:, 1] / max(
        summed[:, 1].sum(), 1e-30
    )
    physical = np.isfinite(opacity[representative]) & (
        opacity[representative] >= KERNEL_ALPHA_CUTOFF
    )
    eligible = np.flatnonzero(physical)
    ranked = eligible[np.argsort(-utility[eligible], kind="stable")]
    return (
        cells,
        representative,
        utility,
        ranked,
        dict(
            valid_teacher_rows=int(valid.sum()),
            exact_sfm_overlap_cells_excluded=overlap,
            novel_cells=len(cells),
            physical_alpha_eligible_cells=len(eligible),
        ),
    )
