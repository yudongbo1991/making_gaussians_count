# Upstream attribution and terms: LICENSE.md
"""Process-local allocator peaks and wall time for the complete training call."""

import json
import os
from pathlib import Path
import time
import torch


class TrainingResources:

    def __init__(self):
        torch.cuda.synchronize()
        self.started = time.perf_counter()
        self.initial_allocated = torch.cuda.memory_allocated()
        self.initial_reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()

    def finish(self, model_path, iterations, anchors):
        torch.cuda.synchronize()
        record = dict(
            completed=True,
            iterations=int(iterations),
            final_anchors=int(anchors),
            training_call_wall_seconds=time.perf_counter() - self.started,
            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(),
            initial_torch_allocated_bytes=self.initial_allocated,
            initial_torch_reserved_bytes=self.initial_reserved,
            physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
            gpu=torch.cuda.get_device_name(),
            scope="Model/scene setup, training loop, in-loop validation and checkpoint writing; excludes subsequent final rendering/metrics.",
            limitation="PyTorch allocator peaks are not total NVML process memory; no historical values are inferred.",
        )
        (Path(model_path) / "training_resources.json").write_text(json.dumps(record, indent=2))
        return record
