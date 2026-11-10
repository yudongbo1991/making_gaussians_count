"""Quality profile for the final analytic method."""

import math

FIELDS = ("psnr", "ssim", "l1", "active")


def joint_profile(fitting, admission):

    def indexed(rows):
        result = {}
        for row in rows:
            count = row["count"]
            if not isinstance(count, int) or count < 0 or count in result:
                raise ValueError("Unique nonnegative integer candidate counts required")
            result[count] = row
        return result

    fits = indexed(fitting)
    adm = indexed(admission)
    if set(fits) != set(adm) or 0 not in fits:
        raise ValueError("Both subsets must measure identical full candidate counts including zero")
    result = []
    reference = None
    for count in sorted(fits):
        left = fits[count]["per_view"]
        right = adm[count]["per_view"]
        if not left or not right:
            raise ValueError("Both camera subsets must be nonempty")
        ids_left = [r["image"] for r in left]
        ids_right = [r["image"] for r in right]
        if len(set(ids_left + ids_right)) != len(left) + len(right):
            raise ValueError("Unique disjoint camera identities required")
        identity = (tuple(ids_left), tuple(ids_right))
        if reference is None:
            reference = identity
        if identity != reference:
            raise ValueError("Camera protocol changed across candidate counts")
        per_view = left + right
        if any((not math.isfinite(float(r[k])) for r in per_view for k in FIELDS)):
            raise ValueError("Finite measured metrics required")
        result.append(
            dict(
                count=count,
                per_view=per_view,
                mean={k: sum((float(r[k]) for r in per_view)) / len(per_view) for k in FIELDS},
                fitting_views=len(left),
                admission_views=len(right),
            )
        )
    return result
