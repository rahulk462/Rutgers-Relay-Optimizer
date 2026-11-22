import itertools
import math

import numpy as np
import pandas as pd
import pytest

from model import get_relay
from solver import (
    InfeasibleError,
    brute_force,
    evaluate,
    hungarian,
    optimize,
    rank_of,
    sensitivity,
)
from times import CostMatrix


def make_cost(values, relay="4x100", names=None):
    values = np.asarray(values, dtype=float)
    r = get_relay(relay)
    names = names or [f"R{i}" for i in range(values.shape[0])]
    times = pd.DataFrame(values, index=names, columns=[leg.name for leg in r.legs])
    sources = pd.DataFrame("split", index=names, columns=times.columns)
    return CostMatrix(r, times, sources)


def random_cost(seed, n_runners):
    rng = np.random.default_rng(seed)
    return make_cost(rng.uniform(9.5, 11.5, size=(n_runners, 4)).round(2))


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("n_runners", [4, 6, 8, 10])
def test_brute_force_and_hungarian_agree(seed, n_runners):
    cost = random_cost(seed, n_runners)
    assert hungarian(cost).total == pytest.approx(brute_force(cost)[0].total)


@pytest.mark.parametrize("seed", range(20))
def test_agree_with_locks_and_exclusions(seed):
    cost = random_cost(seed, 8)
    rng = np.random.default_rng(seed + 1000)
    names = cost.runners
    excluded = set(rng.choice(names, size=2, replace=False))
    rest = [n for n in names if n not in excluded]
    locks = {int(rng.integers(4)): str(rng.choice(rest))}
    bf = brute_force(cost, locks, excluded)
    h = hungarian(cost, locks, excluded)
    assert h.total == pytest.approx(bf[0].total)
    for lineup in (bf[0], h):
        for leg, runner in locks.items():
            assert lineup.runners[leg] == runner
        assert not excluded & set(lineup.runners)


@pytest.mark.parametrize("seed", range(10))
def test_agree_with_missing_times(seed):
    rng = np.random.default_rng(seed)
    values = rng.uniform(9.5, 11.5, size=(7, 4))
    values[rng.random(values.shape) < 0.3] = math.inf
    cost = make_cost(values)
    bf = brute_force(cost)
    if not bf:
        with pytest.raises(InfeasibleError):
            hungarian(cost)
    else:
        assert hungarian(cost).total == pytest.approx(bf[0].total)


def test_brute_force_counts_every_lineup():
    cost = random_cost(0, 8)
    assert len(brute_force(cost)) == 8 * 7 * 6 * 5 == 1680


def test_ranking_is_sorted_and_unique():
    ranking = brute_force(random_cost(3, 6))
    totals = [round(l.total, 6) for l in ranking]
    assert totals == sorted(totals)
    assert len({l.runners for l in ranking}) == len(ranking)


def test_known_answer_leg_specialists():
    # A is the best starter, B only good on leg 2, etc. Fastest overall runner E is beaten
    # by specialists everywhere except leg 4.
    cost = make_cost(
        [
            [10.0, 10.5, 10.5, 10.5],
            [10.5, 9.4, 10.5, 10.5],
            [10.5, 10.5, 9.6, 10.5],
            [10.5, 10.5, 10.5, 9.8],
            [10.2, 9.5, 9.7, 9.5],
        ],
        names=list("ABCDE"),
    )
    best = optimize(cost).best
    assert best.runners == ("A", "B", "C", "E")
    assert best.total == pytest.approx(10.0 + 9.4 + 9.6 + 9.5)


def test_optimize_returns_top_k():
    cost = random_cost(5, 8)
    result = optimize(cost, k=6)
    assert len(result.alternatives) == 5
    assert result.ranked() == brute_force(cost)[:6]


def test_evaluate_and_rank_of_preferred_lineup():
    cost = random_cost(7, 6)
    ranking = brute_force(cost)
    third = ranking[2]
    mine = evaluate(cost, third.runners)
    assert mine.total == pytest.approx(third.total)
    assert rank_of(mine, ranking) <= 3
    assert rank_of(ranking[0], ranking) == 1


def test_evaluate_rejects_bad_lineups():
    cost = random_cost(1, 6)
    with pytest.raises(ValueError):
        evaluate(cost, ["R0", "R0", "R1", "R2"])
    with pytest.raises(ValueError):
        evaluate(cost, ["R0", "R1", "R2"])
    with pytest.raises(ValueError):
        evaluate(cost, ["R0", "R1", "R2", "Nobody"])


def test_infeasible_constraints():
    cost = random_cost(2, 6)
    with pytest.raises(InfeasibleError):
        optimize(cost, locks={0: "R1", 1: "R1"})
    with pytest.raises(InfeasibleError):
        optimize(cost, locks={0: "R1"}, exclusions={"R1"})
    with pytest.raises(InfeasibleError):
        optimize(cost, exclusions={"R0", "R1", "R2"})


def test_sensitivity_margins_are_exact():
    cost = random_cost(11, 7)
    result = optimize(cost)
    for s in sensitivity(result, cost):
        values = cost.times.copy()
        leg = cost.relay.legs[s.leg].name
        bump = s.seconds + 0.01
        values.at[s.runner, leg] += bump if s.in_lineup else -bump
        changed = optimize(CostMatrix(cost.relay, values, cost.sources)).best
        assert changed.runners != result.best.runners
        if s.seconds > 0.02:
            values.at[s.runner, leg] -= 0.02 if s.in_lineup else -0.02
            same = optimize(CostMatrix(cost.relay, values, cost.sources)).best
            assert same.runners == result.best.runners
