"""Train-camera workload feedback; no native/test target or fixed prune fraction."""

import json
from pathlib import Path


def feedback_decision(reference_rows, current_rows):
    """Compare exact same admission-camera active counts, without PSNR selection."""
    reference = {r["image"]: r["active"] for r in reference_rows}
    current = {r["image"]: r["active"] for r in current_rows}
    if len(reference) != len(reference_rows) or len(current) != len(current_rows):
        raise ValueError("Duplicate admission image")
    if not reference or reference.keys() != current.keys():
        raise ValueError("Feedback requires identical nonempty training cameras")
    for count in [*reference.values(), *current.values()]:
        if isinstance(count, bool) or count < 0 or int(count) != count:
            raise ValueError("Active counts must be nonnegative integers")
    before = sum(reference.values())
    after = sum(current.values())
    return dict(
        trigger=after > before,
        reference_sum=before,
        current_sum=after,
        rebound_fraction=(after - before) / before if before else None,
        number_of_training_views=len(reference),
        reference_mean=before / len(reference),
        current_mean=after / len(reference),
    )


def check_and_recover(model, cameras, pipe, background, mode, teacher, iteration, model_path):
    import torch
    import gaussian_renderer as renderer
    from scene.adaptive_recovery import contract_adaptive

    directory = Path(model_path)
    path = directory / "ispg_refresh.json"
    if path.exists():
        raise FileExistsError("Never overwrite a feedback decision")
    first = json.loads((directory / "ispg_25000.json").read_text())
    assert first["iteration"] == 25000 and first["mode"] == mode
    assert first["optimizer_steps"] == 0 and (not first["official_test_used"])
    camera_by_name = {c.image_name: c for c in cameras}
    reference = first["admission_proposed"]["per_view"]
    cpu_rng, cuda_rng = (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())
    initial = model.slot_keep.detach().clone()
    rows = []
    with torch.no_grad():
        for row in reference:
            camera = camera_by_name[row["image"]]
            visible = renderer.prefilter_voxel(camera, model, pipe, background)
            rendered = renderer.render(camera, model, pipe, background, visible_mask=visible)
            rows.append(dict(image=camera.image_name, active=int((rendered["radii"] > 0).sum())))
    decision = feedback_decision(reference, rows)
    decision.update(
        iteration=iteration,
        mode=mode,
        optimizer_steps=0,
        official_test_used=False,
        reference_iteration=25000,
        reference_per_view=reference,
        current_per_view=rows,
        trigger_rule="strict rise in total active count on identical train admission cameras",
        checkpoint_iteration_is_fixed_engineering_setting=True,
        stage2_report=None,
    )
    if decision["trigger"]:
        output = directory / "ispg_29000.json"
        contract_adaptive(model, cameras, pipe, background, mode, teacher, iteration, str(output))
        stage2 = json.loads(output.read_text())
        if torch.any(model.slot_keep & ~initial):
            raise RuntimeError("Feedback regenerated a previously removed slot")
        removed = int((initial & ~model.slot_keep).sum())
        if removed != stage2["accepted_count"]:
            raise RuntimeError("Stage2 inherited-mask accounting differs")
        decision.update(stage2_report=str(output), newly_removed_slots=removed)
    else:
        assert torch.equal(initial, model.slot_keep)
        decision["newly_removed_slots"] = 0
    if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(
        cuda_rng, torch.cuda.get_rng_state()
    ):
        raise RuntimeError("Feedback perturbed global random streams")
    decision["global_rng_unchanged"] = True
    path.write_text(json.dumps(decision, indent=2))
    print(
        "Train-only rebound feedback:",
        decision["trigger"],
        decision["newly_removed_slots"],
        flush=True,
    )
