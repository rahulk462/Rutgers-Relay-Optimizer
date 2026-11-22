import math
from pathlib import Path

import pandas as pd
import pytest

from model import RELAYS, get_relay
from solver import optimize
from times import (
    DEFAULT_ADJUSTMENTS,
    Adjustment,
    build_cost_matrix,
    clean_roster,
    format_time,
    load_roster,
    parse_time,
)

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data.csv"


@pytest.mark.parametrize(
    "text, seconds",
    [("10.85", 10.85), ("4:05.3", 245.3), ("1:49.10", 109.1), (" 47.9 ", 47.9), ("1:02:03", 3723.0), (9.9, 9.9)],
)
def test_parse_time(text, seconds):
    assert parse_time(text) == pytest.approx(seconds)


@pytest.mark.parametrize("blank", ["", "  ", None, "nan", "-", float("nan")])
def test_parse_blank(blank):
    assert math.isnan(parse_time(blank))


def test_parse_garbage():
    with pytest.raises(ValueError):
        parse_time("fast")


def test_format_time():
    assert format_time(10.854) == "10.85"
    assert format_time(245.3) == "4:05.30"
    assert format_time(math.inf) == "—"


def test_adjustment():
    assert Adjustment(1.0, -0.9).apply(10.8) == pytest.approx(9.9)
    assert Adjustment(1.02, 0.0).apply(100) == pytest.approx(102)


def test_every_leg_type_has_a_default():
    for relay in RELAYS.values():
        for leg in relay.legs:
            assert leg.leg_type.key in DEFAULT_ADJUSTMENTS


def roster(**cols):
    return clean_roster(pd.DataFrame({"runner": ["A", "B"], **cols}))


def test_open_time_conversion_uses_leg_type():
    r = roster(open_100=["10.80", "11.00"])
    cost = build_cost_matrix(r, get_relay("4x100"))
    a = cost.times.loc["A"]
    assert a["Leg 1"] == pytest.approx(10.80 + 0.10)
    assert a["Leg 2"] == pytest.approx(10.80 - 0.90)
    assert a["Leg 3"] == pytest.approx(10.80 - 0.75)
    assert a["Leg 4"] == pytest.approx(10.80 - 0.90)
    assert (cost.sources == "converted").all().all()


def test_split_overrides_conversion():
    r = roster(open_100=["10.80", "11.00"], **{"4x100_leg3": ["9.70", ""]})
    cost = build_cost_matrix(r, get_relay("4x100"))
    assert cost.times.at["A", "Leg 3"] == pytest.approx(9.70)
    assert cost.sources.at["A", "Leg 3"] == "split"
    assert cost.times.at["B", "Leg 3"] == pytest.approx(11.00 - 0.75)


def test_custom_adjustments():
    r = roster(open_400=["48.0", "49.0"])
    adj = {"400_lanes": Adjustment(1.0, 1.0)}
    cost = build_cost_matrix(r, get_relay("4x400"), adj)
    assert cost.times.at["A", "Leg 1"] == pytest.approx(49.0)
    assert cost.times.at["A", "Leg 2"] == pytest.approx(48.0 - 0.70)  # default kept


def test_dmr_uses_each_distance():
    r = roster(open_1200=["3:00", ""], open_400=["50.0", "49.0"], open_800=["1:52", ""], open_1600=["", "4:05"])
    cost = build_cost_matrix(r, get_relay("DMR"))
    assert cost.times.at["A", "Leg 1"] == pytest.approx(180.0)
    assert cost.times.at["B", "Leg 2"] == pytest.approx(49.0 - 0.70)
    assert cost.times.at["A", "Leg 3"] == pytest.approx(112.0 - 0.50)
    assert cost.times.at["B", "Leg 4"] == pytest.approx(245.0 - 0.50)
    assert math.isinf(cost.times.at["A", "Leg 4"])
    assert cost.sources.at["A", "Leg 4"] == "missing"


def test_runner_subset_and_unknown_runner():
    r = roster(open_100=["10.8", "11.0"])
    assert build_cost_matrix(r, get_relay("4x100"), runners=["B"]).runners == ["B"]
    with pytest.raises(ValueError):
        build_cost_matrix(r, get_relay("4x100"), runners=["Z"])


def test_duplicate_runner_rejected():
    with pytest.raises(ValueError):
        clean_roster(pd.DataFrame({"runner": ["A", "A"], "open_100": ["10", "11"]}))


@pytest.mark.parametrize("relay", list(RELAYS))
def test_sample_data_solves_every_relay(relay):
    cost = build_cost_matrix(load_roster(SAMPLE), get_relay(relay))
    result = optimize(cost)
    assert len(set(result.best.runners)) == 4
    assert math.isfinite(result.best.total)


def test_all_major_relays_defined():
    assert set(RELAYS) == {"4x100", "4x200", "4x400", "4x800", "SHR", "SMR", "DMR"}
    for relay in RELAYS.values():
        assert relay.n_legs == 4


def test_4x200_and_4x800_conversion():
    r = roster(open_200=["21.50", "22.00"], open_800=["1:50.0", "1:52.0"])
    c200 = build_cost_matrix(r, get_relay("4x200")).times
    assert c200.at["A", "Leg 1"] == pytest.approx(21.50 + 0.10)
    assert c200.at["A", "Leg 3"] == pytest.approx(21.50 - 0.60)
    c800 = build_cost_matrix(r, get_relay("4x800")).times
    assert c800.at["B", "Leg 1"] == pytest.approx(112.0)
    assert c800.at["B", "Leg 4"] == pytest.approx(112.0 - 0.50)


def test_shuttle_hurdles_use_hurdle_times():
    r = roster(open_hurdles=["13.90", "14.30"], open_100=["10.5", "10.6"])
    cost = build_cost_matrix(r, get_relay("SHR")).times
    assert cost.at["A", "Leg 1"] == pytest.approx(13.90)
    assert cost.at["A", "Leg 2"] == pytest.approx(13.90 + 0.10)  # touch-off, no flying start


def test_smr_uses_200_200_400_800():
    r = roster(open_200=["21.5", ""], open_400=["48.0", ""], open_800=["", "1:51.0"])
    relay = get_relay("SMR")
    assert [leg.distance_m for leg in relay.legs] == [200, 200, 400, 800]
    cost = build_cost_matrix(r, relay).times
    assert cost.at["A", "Leg 1"] == pytest.approx(21.6)
    assert cost.at["A", "Leg 2"] == pytest.approx(20.9)
    assert cost.at["A", "Leg 3"] == pytest.approx(47.3)
    assert cost.at["B", "Leg 4"] == pytest.approx(110.5)
    assert math.isinf(cost.at["A", "Leg 4"])
