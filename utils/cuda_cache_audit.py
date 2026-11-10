# Upstream attribution and terms: LICENSE.md
"""Release only unused CUDA allocator blocks, without changing model policy.

No peak-stat reset, resizing, renderer changes, RNG calls or quantity cap.
The audit records allocator memory, not total NVML process memory. GPU runtime
equivalence and the user15GB process-peak target still require measurement.
"""

import json
from pathlib import Path


def release_cuda_cache(path, stage, cuda=None):
    if cuda is None:
        from torch import cuda
    cuda.synchronize()
    allocated = cuda.memory_allocated()
    reserved = cuda.memory_reserved()
    peak_allocated = cuda.max_memory_allocated()
    peak_reserved = cuda.max_memory_reserved()
    cuda.empty_cache()
    cuda.synchronize()
    after_allocated = cuda.memory_allocated()
    after_reserved = cuda.memory_reserved()
    if after_allocated != allocated or after_reserved > reserved:
        raise RuntimeError("Cache release changed live allocation or increased reserve")
    record = dict(
        stage=stage,
        allocated_before=allocated,
        allocated_after=after_allocated,
        reserved_before=reserved,
        reserved_after=after_reserved,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        peaks_reset=False,
        model_quantity_cap=None,
        scope="PyTorch allocator only; not a15GB total-process-peak proof",
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    return record
