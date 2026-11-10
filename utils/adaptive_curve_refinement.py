# Upstream attribution and terms: LICENSE.md
"""Refine measured discrete knee, not a fixed retirement fraction.

Always pool the full measured range, preserving the original endpoints.
Stop at integer resolution, or a search budget
derived from the bit length of the actual candidate range. No test views.
"""


def refine_profile(curve, admission_before, trials, evaluate_count, pool, knee):
    curve = list(curve)
    trials = list(trials)
    counts = sorted((r["count"] for r in curve))
    if not counts or counts[0] != 0 or len(set(counts)) != len(counts):
        raise ValueError("Unique full measured profile including zero required")
    if sorted((r["count"] for r in trials)) != counts[1:]:
        raise ValueError("Every positive count needs measured admission")
    endpoint = counts[-1]

    def selected():
        profile = pool(curve, [dict(count=0, **admission_before)] + trials)
        return (profile, profile[knee([r["mean"] for r in profile])]["count"])

    profile, winner = selected()
    rounds = []
    stop = "range_derived_search_budget"
    budget = max(1, endpoint.bit_length())
    for _ in range(budget):
        counts = sorted((r["count"] for r in curve))
        index = counts.index(winner)
        neighbors = counts[max(0, index - 1) : index] + counts[index + 1 : index + 2]
        proposals = sorted({(winner + n) // 2 for n in neighbors} - {*counts})
        if not proposals:
            stop = "integer_resolution"
            break
        before = winner
        for count in proposals:
            fitting, admission = evaluate_count(count)
            curve.append(dict(count=count, **fitting))
            trials.append(dict(count=count, **admission))
        curve.sort(key=lambda r: r["count"])
        trials.sort(key=lambda r: r["count"])
        if curve[0]["count"] != 0 or curve[-1]["count"] != endpoint:
            raise ValueError("Refinement changed global range")
        profile, winner = selected()
        rounds.append(dict(winner_before=before, evaluated_counts=proposals, winner_after=winner))
    return (
        curve,
        trials,
        profile,
        dict(
            rounds=rounds,
            stop_reason=stop,
            original_upper=endpoint,
            search_budget_from_candidate_bit_length=budget,
            fixed_retirement_fraction=None,
            quantity_target=None,
            full_range_normalization_preserved=True,
            accepted=winner,
            caveat="Finite measured search, not proof of a globally optimal continuous quality/count curve.",
        ),
    )
