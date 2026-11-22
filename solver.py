"""Find the fastest lineup: match runners to legs so the total expected time is smallest.

This is an assignment problem. Two solvers are provided and should always agree:

* :func:`brute_force` checks every ordered lineup (8 runners -> 1,680, 10 -> 5,040), which
  also gives a full ranking for the "next fastest" alternatives.
* :func:`hungarian` uses ``scipy.optimize.linear_sum_assignment``.

Coach constraints:

* ``locks``: ``{leg_index: runner}`` pins a runner to a leg (0-based leg index).
* ``exclusions``: runners who can't run at all.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from times import CostMatrix

TOL = 1e-9


class InfeasibleError(ValueError):
    """No lineup satisfies the constraints (too few runners, or locks conflict)."""


@dataclass(frozen=True)
class Lineup:
    runners: tuple[str, ...]  # runners[j] runs leg j
    leg_times: tuple[float, ...]

    @property
    def total(self) -> float:
        return float(sum(self.leg_times))


@dataclass(frozen=True)
class Sensitivity:
    runner: str
    leg: int  # 0-based leg index
    in_lineup: bool
    seconds: float
    # in_lineup=True: how much slower this runner's leg can get before the best lineup changes.
    # in_lineup=False: how much faster the runner must be on this leg to make the best lineup.


@dataclass
class Result:
    best: Lineup
    alternatives: list[Lineup]  # next fastest, best first, excluding ``best``
    n_feasible: int
    all_lineups: list[Lineup] = field(repr=False)

    def ranked(self) -> list[Lineup]:
        return [self.best, *self.alternatives]


def _normalize(cost: CostMatrix, locks: dict[int, str] | None, exclusions) -> tuple[dict[int, str], set[str]]:
    locks = dict(locks or {})
    exclusions = set(exclusions or ())
    runners = set(cost.runners)
    n_legs = cost.relay.n_legs
    for leg, runner in locks.items():
        if not 0 <= leg < n_legs:
            raise ValueError(f"Lock on leg index {leg}, but {cost.relay.name} has {n_legs} legs")
        if runner not in runners:
            raise InfeasibleError(f"{runner} is locked in but isn't an available runner")
        if runner in exclusions:
            raise InfeasibleError(f"{runner} is both locked in and excluded")
    locked = list(locks.values())
    dupes = {r for r in locked if locked.count(r) > 1}
    if dupes:
        raise InfeasibleError(f"{', '.join(sorted(dupes))} is locked to more than one leg")
    unknown = exclusions - runners
    if unknown:
        raise ValueError(f"Excluded runners not in the pool: {', '.join(sorted(unknown))}")
    return locks, exclusions


def constrained_matrix(cost: CostMatrix, locks=None, exclusions=None) -> np.ndarray:
    """Cost values with constraints applied as infinite costs."""
    locks, exclusions = _normalize(cost, locks, exclusions)
    c = cost.values().copy()
    idx = {r: i for i, r in enumerate(cost.runners)}
    for r in exclusions:
        c[idx[r], :] = math.inf
    for leg, r in locks.items():
        i = idx[r]
        keep = c[i, leg]
        c[:, leg] = math.inf
        c[i, :] = math.inf
        c[i, leg] = keep
    return c


def _lineup(cost: CostMatrix, c: np.ndarray, rows) -> Lineup:
    return Lineup(
        tuple(cost.runners[i] for i in rows),
        tuple(float(c[i, j]) for j, i in enumerate(rows)),
    )


def brute_force(cost: CostMatrix, locks=None, exclusions=None) -> list[Lineup]:
    """Every feasible lineup, fastest first."""
    c = constrained_matrix(cost, locks, exclusions)
    n_runners, n_legs = c.shape
    candidates = [i for i in range(n_runners) if np.isfinite(c[i]).any()]
    cols = np.arange(n_legs)
    found = []
    for rows in itertools.permutations(candidates, n_legs):
        total = c[list(rows), cols].sum()
        if math.isfinite(total):
            found.append((total, rows))
    found.sort(key=lambda t: round(t[0], 6))  # stable, so ties keep roster order
    return [_lineup(cost, c, rows) for _, rows in found]


def hungarian(cost: CostMatrix, locks=None, exclusions=None) -> Lineup:
    """The single fastest lineup via the Hungarian algorithm."""
    c = constrained_matrix(cost, locks, exclusions)
    n_runners, n_legs = c.shape
    if n_runners < n_legs:
        raise InfeasibleError(f"Need {n_legs} runners, only {n_runners} available")
    finite = c[np.isfinite(c)]
    if finite.size == 0:
        raise InfeasibleError("No runner has a time for any leg")
    big = (abs(finite).max() + 1) * (n_legs + 1) * 10
    rows, legs = linear_sum_assignment(np.where(np.isfinite(c), c, big))
    if not np.isfinite(c[rows, legs]).all():
        raise InfeasibleError("No lineup covers every leg with the current runners and constraints")
    by_leg = [int(r) for _, r in sorted(zip(legs, rows))]
    return _lineup(cost, c, by_leg)


def optimize(cost: CostMatrix, k: int = 6, locks=None, exclusions=None) -> Result:
    """Best lineup plus the next ``k - 1`` alternatives. Checks brute force against Hungarian."""
    ranking = brute_force(cost, locks, exclusions)
    if not ranking:
        raise InfeasibleError("No lineup covers every leg with the current runners and constraints")
    best_h = hungarian(cost, locks, exclusions)
    if abs(best_h.total - ranking[0].total) > 1e-6:
        raise RuntimeError(
            f"Solvers disagree: brute force {ranking[0].total:.3f}, Hungarian {best_h.total:.3f}"
        )
    return Result(ranking[0], ranking[1:k], len(ranking), ranking)


def evaluate(cost: CostMatrix, runners: list[str] | tuple[str, ...]) -> Lineup:
    """Predicted times for a specific lineup (e.g. the coach's preferred one)."""
    if len(runners) != cost.relay.n_legs:
        raise ValueError(f"A {cost.relay.name} lineup needs {cost.relay.n_legs} runners")
    if len(set(runners)) != len(runners):
        raise ValueError("A runner can only run one leg")
    idx = {r: i for i, r in enumerate(cost.runners)}
    unknown = [r for r in runners if r not in idx]
    if unknown:
        raise ValueError(f"Not available: {', '.join(unknown)}")
    return _lineup(cost, cost.values(), [idx[r] for r in runners])


def rank_of(lineup: Lineup, ranking: list[Lineup]) -> int:
    """1-based place of ``lineup`` among ``ranking`` (ties share the better place)."""
    return 1 + sum(1 for l in ranking if l.total < lineup.total - TOL)


def sensitivity(result: Result, cost: CostMatrix) -> list[Sensitivity]:
    """How much each runner's time would have to change to change the best lineup.

    For a runner in the best lineup: how many seconds slower their leg can get before some
    other lineup is faster. For a runner left out: how many seconds faster they'd need to be
    on their most promising leg to get into the best lineup.
    """
    best = result.best
    out: list[Sensitivity] = []
    for j, r in enumerate(best.runners):
        rivals = [l.total for l in result.all_lineups if l.runners[j] != r]
        if rivals:
            out.append(Sensitivity(r, j, True, round(min(rivals) - best.total, 6)))
    for r in cost.runners:
        if r in best.runners:
            continue
        options = []
        for j in range(cost.relay.n_legs):
            totals = [l.total for l in result.all_lineups if l.runners[j] == r]
            if totals:
                options.append((round(min(totals) - best.total, 6), j))
        if options:
            gap, j = min(options)
            out.append(Sensitivity(r, j, False, gap))
    return out
