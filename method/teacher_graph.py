"""Five-teacher support allocation graph."""

from functools import lru_cache
import hashlib
import numpy as np
from method.teacher_ellipse import EllipticalTeacherSurface
from method.teacher_overlap import ellipse_incidence
from method.io import digest


@lru_cache(maxsize=1)
def _surface(path, sha256):
    assert digest(path) == sha256
    return EllipticalTeacherSurface(path)


def transition(surface, names):
    names = np.asarray(names)
    if names.ndim != 1 or not np.issubdtype(names.dtype, np.integer):
        raise ValueError("One integral teacher identity per source group required")
    if (names < 0).any() or (names >= len(surface.xyz)).any():
        raise ValueError("Actual teacher identities required")
    ids = np.empty((len(names), 5), dtype=np.int64)
    weights = np.empty((len(names), 5), dtype=np.float64)
    fallback = 0
    for start in range(0, len(names), 16384):
        stop = min(start + 16384, len(names))
        targets, scores = ellipse_incidence(surface, surface.xyz[names[start:stop]])
        totals = scores.sum(1)
        for row in np.flatnonzero(totals == 0):
            source = names[start + row]
            matches = np.flatnonzero(targets[row] == source)
            if not len(matches):
                targets[row, 0] = source
                index = 0
            else:
                index = int(matches[0])
            scores[row] = 0.0
            scores[row, index] = 1.0
            fallback += 1
        scores /= scores.sum(1, keepdims=True)
        ids[start:stop] = targets
        weights[start:stop] = scores
    return (
        ids,
        weights,
        dict(
            zero_kernel_identity_fallback_rows=fallback,
            transition_sha256=hashlib.sha256(ids.tobytes() + weights.tobytes()).hexdigest(),
            transition_rows=len(names),
            teacher_neighbors=5,
        ),
    )
