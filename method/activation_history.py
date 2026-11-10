"""Activation history for the final analytic method."""

import torch


class ActivationHistory:

    def __init__(self, keep, training_images, start_iteration=15000):
        self.active = torch.zeros_like(keep, dtype=torch.bool)
        self.training_images = frozenset(training_images)
        if not self.training_images:
            raise ValueError("Training image set required")
        self.visits = {name: 0 for name in self.training_images}
        self.start_iteration = start_iteration
        self.last_iteration = start_iteration
        self.render_observations = 0

    @torch.no_grad()
    def observe(self, visible, selection, image, iteration):
        if iteration <= self.last_iteration:
            raise ValueError("Ordinary observations must be ordered")
        if image not in self.visits:
            raise ValueError("Only declared training cameras permitted")
        if visible.dtype != torch.bool or selection.dtype != torch.bool:
            raise ValueError("Boolean renderer masks required")
        if visible.numel() != self.active.shape[0]:
            raise ValueError("Post-core anchor identity changed")
        parents = visible.nonzero().flatten()
        slots = self.active.shape[1]
        if selection.numel() != len(parents) * slots:
            raise ValueError("Renderer selection layout differs")
        selected = (
            parents[:, None] * slots + torch.arange(slots, device=visible.device)
        ).flatten()[selection]
        self.active.flatten()[selected] = True
        self.visits[image] += 1
        self.render_observations += 1
        self.last_iteration = iteration

    def metadata(self):
        return dict(
            start_iteration=self.start_iteration,
            last_iteration=self.last_iteration,
            ordinary_render_observations=self.render_observations,
            distinct_training_images_seen=sum((v > 0 for v in self.visits.values())),
            declared_training_images=len(self.visits),
            all_training_images_seen=all((v > 0 for v in self.visits.values())),
            per_training_image_visits=dict(sorted(self.visits.items())),
            extra_render_calls=0,
            extra_optimizer_steps=0,
            official_test_used=False,
        )
