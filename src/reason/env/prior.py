"""Paper equation (13): discounted mean log prior for a macro action."""

import math


def macro_action_log_score(step_logprobs, eta: float, eps: float) -> float:
    if not 0 < eta <= 1:
        raise ValueError("eta must be in (0, 1]")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if not step_logprobs:
        raise ValueError("macro action must contain at least one transition")
    weights = [eta ** index for index in range(len(step_logprobs))]
    log_eps = math.log(eps)
    terms = []
    for weight, log_p in zip(weights, step_logprobs):
        log_p = float(log_p)
        if not math.isfinite(log_p):
            if log_p != -math.inf:
                raise ValueError("step log prior must be finite or -inf")
        upper = max(log_p, log_eps)
        terms.append(weight * (upper + math.log1p(math.exp(min(log_p, log_eps) - upper))))
    return math.fsum(terms) / math.fsum(weights)
