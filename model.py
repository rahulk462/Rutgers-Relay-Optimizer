"""Relay definitions: which legs each relay has, how far each leg is, and what kind of leg it is.

The leg type is what drives the time conversion in ``times.py``: two legs of the same
distance can still differ (a 100 m from the blocks on a curve is slower than a flying 100 m
on a straight), so each distinct situation gets its own leg type.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LegType:
    key: str
    label: str
    open_event: str  # the open-race event whose time this leg is converted from, e.g. "100"
    description: str


@dataclass(frozen=True)
class Leg:
    number: int  # 1-based
    distance_m: int
    leg_type: LegType

    @property
    def name(self) -> str:
        return f"Leg {self.number}"


@dataclass(frozen=True)
class Relay:
    key: str
    name: str
    legs: tuple[Leg, ...]

    @property
    def n_legs(self) -> int:
        return len(self.legs)

    def split_column(self, leg_number: int) -> str:
        """Column name holding a runner's known split for this relay leg, e.g. ``4x100_leg3``."""
        return f"{self.key}_leg{leg_number}"


LEG_TYPES: dict[str, LegType] = {
    lt.key: lt
    for lt in [
        # 4x100
        LegType("100_blocks_curve", "4x100 leg 1: blocks, curve", "100",
                "Starts from the blocks and runs the first curve."),
        LegType("100_flying_straight", "4x100 legs 2 and 4: flying, straight", "100",
                "Flying start, runs a straight."),
        LegType("100_flying_curve", "4x100 leg 3: flying, curve", "100",
                "Flying start, runs the second curve."),
        # 4x200
        LegType("200_blocks", "4x200 leg 1: blocks, staggered curve", "200",
                "Starts from the blocks on a staggered curve."),
        LegType("200_flying", "4x200 legs 2 to 4: flying", "200",
                "Flying start, a curve and a straight."),
        # 4x400
        LegType("400_lanes", "4x400 leg 1: blocks, all in lanes", "400",
                "Starts from the blocks and stays in lanes the whole lap."),
        LegType("400_break_in", "4x400 leg 2: flying, cuts in", "400",
                "Flying start, runs the first turn in lanes, then cuts in."),
        LegType("400_traffic", "4x400 legs 3 and 4: flying, in traffic", "400",
                "Flying start from the inside lane, deals with traffic."),
        # 4x800
        LegType("800_lead", "4x800 leg 1: standing start", "800",
                "Standing (waterfall or alley) start, 800 m."),
        LegType("800_flying", "4x800 legs 2 to 4: flying", "800",
                "Flying start, 800 m in traffic."),
        # Shuttle hurdle relay
        LegType("shuttle_blocks", "Shuttle hurdles leg 1: blocks", "hurdles",
                "Starts from the blocks, a full hurdle race (110 mH men, 100 mH women)."),
        LegType("shuttle_touch", "Shuttle hurdles legs 2 to 4: touch-off start", "hurdles",
                "Starts when the incoming runner crosses the line (no baton, no flying start), "
                "running the other way down the straight."),
        # SMR (200-200-400-800)
        LegType("smr_200_blocks", "SMR leg 1: 200 m, blocks", "200", "Starts from the blocks, 200 m."),
        LegType("smr_200_flying", "SMR leg 2: 200 m, flying", "200", "Flying start, 200 m."),
        LegType("smr_400", "SMR leg 3: 400 m, flying", "400", "Flying start, 400 m."),
        LegType("smr_800", "SMR leg 4: 800 m, flying", "800", "Flying start, 800 m."),
        # DMR
        LegType("dmr_1200", "DMR leg 1: 1200 m, standing start", "1200",
                "Standing start, 1200 m."),
        LegType("dmr_400", "DMR leg 2: 400 m, flying", "400", "Flying start, 400 m."),
        LegType("dmr_800", "DMR leg 3: 800 m, flying", "800", "Flying start, 800 m."),
        LegType("dmr_1600", "DMR leg 4: 1600 m, flying", "1600", "Flying start, 1600 m."),
    ]
}


def _relay(key: str, name: str, legs: list[tuple[int, str]]) -> Relay:
    return Relay(key, name, tuple(Leg(i + 1, d, LEG_TYPES[t]) for i, (d, t) in enumerate(legs)))


RELAYS: dict[str, Relay] = {
    r.key: r
    for r in [
        _relay("4x100", "4x100 m", [
            (100, "100_blocks_curve"),
            (100, "100_flying_straight"),
            (100, "100_flying_curve"),
            (100, "100_flying_straight"),
        ]),
        _relay("4x200", "4x200 m", [
            (200, "200_blocks"),
            (200, "200_flying"),
            (200, "200_flying"),
            (200, "200_flying"),
        ]),
        _relay("4x400", "4x400 m", [
            (400, "400_lanes"),
            (400, "400_break_in"),
            (400, "400_traffic"),
            (400, "400_traffic"),
        ]),
        _relay("4x800", "4x800 m", [
            (800, "800_lead"),
            (800, "800_flying"),
            (800, "800_flying"),
            (800, "800_flying"),
        ]),
        _relay("SHR", "Shuttle hurdle relay", [
            (110, "shuttle_blocks"),
            (110, "shuttle_touch"),
            (110, "shuttle_touch"),
            (110, "shuttle_touch"),
        ]),
        _relay("SMR", "Sprint medley (SMR, 200-200-400-800)", [
            (200, "smr_200_blocks"),
            (200, "smr_200_flying"),
            (400, "smr_400"),
            (800, "smr_800"),
        ]),
        _relay("DMR", "Distance medley (DMR)", [
            (1200, "dmr_1200"),
            (400, "dmr_400"),
            (800, "dmr_800"),
            (1600, "dmr_1600"),
        ]),
    ]
}

# Open events a roster can list times for (columns ``open_<event>``).
# "hurdles" is the runner's open hurdle time: 110 mH for men, 100 mH for women.
OPEN_EVENTS: tuple[str, ...] = ("100", "200", "400", "800", "1200", "1600", "hurdles")


def open_column(event: str) -> str:
    return f"open_{event}"


def get_relay(key: str) -> Relay:
    try:
        return RELAYS[key]
    except KeyError:
        raise ValueError(f"Unknown relay {key!r}; choose from {sorted(RELAYS)}") from None
