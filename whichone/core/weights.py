"""Weight elicitation for Stage 2: AHP for the deliberative channel, and the
loading matrix that converts task-criterion intensities to resource weights.

The loading matrix is a governance artefact, not code: organisations override
it with their own versioned copy (see catalogue snapshots / API config). The
default maps each sensitivity onto one resource.
"""
from __future__ import annotations

import numpy as np

# Saaty's Random Index for CR calculation, indexed by matrix order n.
_RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32,
       8: 1.41, 9: 1.45, 10: 1.49}


def ahp_weights(pairwise: np.ndarray) -> tuple[np.ndarray, float]:
    """Weights from a positive reciprocal pairwise-comparison matrix.

    Uses the normalised geometric mean of rows (Saaty's geometric-mean method) and
    returns (weights, consistency_ratio). Callers should reject CR >= 0.10.
    """
    a = np.asarray(pairwise, dtype=float)
    n = a.shape[0]
    if a.shape != (n, n):
        raise ValueError("pairwise matrix must be square")
    if (a <= 0).any():
        raise ValueError("pairwise matrix must be strictly positive")

    g = np.prod(a, axis=1) ** (1.0 / n)
    w = g / g.sum()

    lam_max = float(np.mean((a @ w) / w))
    ci = (lam_max - n) / (n - 1) if n > 2 else 0.0
    ri = _RI.get(n, 1.49)
    cr = 0.0 if ri == 0 else ci / ri
    return w, cr


# Default mapping from task-profile criteria to resource weights (paper,
# Stage 2, automated way): each sensitivity loads onto one resource. Context,
# safety and data governance generate hard constraints, never weights.
DEFAULT_LOADING: dict[str, dict[str, float]] = {
    "carbon_sensitivity": {"energy": 1.0},
    "cost_sensitivity":   {"cost": 1.0},
    "time_sensitivity":   {"time": 1.0},
}


def intensities_to_weights(intensities: dict[str, int],
                           objective_ids: list[str],
                           loading: dict[str, dict[str, float]] | None = None
                           ) -> dict[str, float]:
    """Map ordinal criterion intensities (1-5) to a normalised weight vector.

    weight_j is proportional to the intensity-weighted sum of loadings onto
    objective j. Unknown criteria are rejected loudly rather than ignored,
    since a silently dropped criterion is a mis-elicitation.
    """
    loading = loading if loading is not None else DEFAULT_LOADING
    unknown = set(intensities) - set(loading)
    if unknown:
        raise ValueError(f"unknown criteria: {sorted(unknown)}")

    raw = {obj: 0.0 for obj in objective_ids}
    for crit, intensity in intensities.items():
        if not 1 <= int(intensity) <= 5:
            raise ValueError(f"intensity for {crit} must be 1..5")
        for obj, load in loading[crit].items():
            if obj not in raw:
                raise ValueError(f"loading matrix targets unknown objective {obj}")
            raw[obj] += intensity * load

    total = sum(raw.values())
    if total == 0:
        raise ValueError("intensities produced an all-zero weight vector")
    return {obj: v / total for obj, v in raw.items()}


def validate_weights(weights: dict[str, float], objective_ids: list[str]) -> dict[str, float]:
    """Check coverage and renormalise explicit weights to sum to one."""
    missing = set(objective_ids) - set(weights)
    extra = set(weights) - set(objective_ids)
    if missing or extra:
        raise ValueError(f"weights must cover exactly the objective set; "
                         f"missing={sorted(missing)} extra={sorted(extra)}")
    if any(w < 0 for w in weights.values()):
        raise ValueError("weights must be non-negative")
    total = sum(weights.values())
    if total == 0:
        raise ValueError("weights sum to zero")
    return {k: v / total for k, v in weights.items()}
