"""Prepared-input serialization and exact integer voxel occupancy."""

import hashlib
from pathlib import Path
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def xyz(vertex):
    return np.stack([vertex[x] for x in ["x", "y", "z"]], axis=1).astype(np.float32)


def quantize(points, width):
    return np.unique(np.rint(points / np.float32(width)).astype(np.int64), axis=0)


def keys(cells):
    return np.ascontiguousarray(cells).view(np.dtype((np.void, 24))).reshape(-1)


def save_prior(path, cells, width):
    path.mkdir()
    anchors = cells.astype(np.float32) * np.float32(width)
    replay = quantize(anchors, width)
    assert np.array_equal(replay, cells), "CPU initializer changed selected cells"
    np.savez_compressed(
        path / "prior.npz",
        anchors=anchors,
        world_offsets=np.zeros((len(anchors), 10, 3), np.float32),
        voxel=np.float32(width),
    )
    return dict(
        anchors=len(cells),
        prior=str(path / "prior.npz"),
        sha256=digest(path / "prior.npz"),
        CPU_revoxelization_exact=True,
    )


def config_namespace(path):
    import ast

    node = ast.parse(Path(path).read_text(), mode="eval").body
    if (
        not isinstance(node, ast.Call)
        or not isinstance(node.func, ast.Name)
        or node.func.id != "Namespace"
        or node.args
    ):
        raise ValueError("Expected a literal Namespace configuration")
    return {k.arg: ast.literal_eval(k.value) for k in node.keywords}
