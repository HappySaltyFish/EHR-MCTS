"""Type-only prefix kernel for macro actions, paper equation (8)."""


def step_kernel(e1, e2):
    return float(int(e1["type"]) == int(e2["type"]))


def prefix_kernel(steps_a, steps_b, ell):
    if ell < 1 or ell > min(len(steps_a), len(steps_b)):
        return 0.0
    return float(all(step_kernel(steps_a[j], steps_b[j]) for j in range(ell)))


def segment_kernel(steps_a, steps_b, prefix_lambda=1.0):
    """Discounted common-type prefixes / all prefixes of the longer action."""
    if not 0 < prefix_lambda <= 1:
        raise ValueError("prefix_lambda must be in (0, 1]")
    shortest = min(len(steps_a), len(steps_b))
    longest = max(len(steps_a), len(steps_b))
    if shortest == 0:
        return 0.0
    numerator = sum(
        prefix_lambda ** (ell - 1) * prefix_kernel(steps_a, steps_b, ell)
        for ell in range(1, shortest + 1)
    )
    denominator = sum(prefix_lambda ** (ell - 1) for ell in range(1, longest + 1))
    return numerator / denominator
