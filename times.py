"""Build the cost matrix c(i, j): runner i's expected time on leg j.

For each runner and leg:

1. If the roster has a known split for that exact relay leg (column ``<relay>_leg<n>``,
   e.g. ``4x100_leg3``), use it.
2. Otherwise convert the runner's open time for the leg's event (column ``open_<event>``,
   e.g. ``open_100``) with the leg type's adjustment: ``expected = open * factor + offset``.
3. If neither is known, the runner can't be placed on that leg (the cost is infinite).

Times may be given as seconds (``10.85``) or as ``m:ss.xx`` (``4:05.3``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from model import LEG_TYPES, Relay, open_column


@dataclass(frozen=True)
class Adjustment:
    factor: float = 1.0
    offset: float = 0.0  # seconds

    def apply(self, open_time: float) -> float:
        return open_time * self.factor + self.offset


# Starting guesses, meant to be edited by the coach. A flying start is worth most of a second
# over 100 m; curves and traffic give some of that back.
DEFAULT_ADJUSTMENTS: dict[str, Adjustment] = {
    "100_blocks_curve": Adjustment(1.0, 0.10),
    "100_flying_straight": Adjustment(1.0, -0.90),
    "100_flying_curve": Adjustment(1.0, -0.75),
    "400_lanes": Adjustment(1.0, 0.20),
    "400_break_in": Adjustment(1.0, -0.70),
    "400_traffic": Adjustment(1.0, -0.50),
    "dmr_1200": Adjustment(1.0, 0.0),
    "dmr_400": Adjustment(1.0, -0.70),
    "dmr_800": Adjustment(1.0, -0.50),
    "dmr_1600": Adjustment(1.0, -0.50),
}
assert set(DEFAULT_ADJUSTMENTS) == set(LEG_TYPES)


def parse_time(value) -> float:
    """Parse a time cell into seconds. Blank / missing values become NaN."""
    if value is None:
        return math.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "-", "na", "n/a"}:
        return math.nan
    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError(f"Can't read time {value!r}")
    try:
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60 + float(part)
    except ValueError:
        raise ValueError(f"Can't read time {value!r}") from None
    return seconds


def format_time(seconds: float) -> str:
    """Seconds to a display string: ``10.85`` or ``4:05.30``."""
    if seconds is None or not math.isfinite(seconds):
        return "—"
    if seconds < 60:
        return f"{seconds:.2f}"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}:{rest:05.2f}"


def load_roster(source) -> pd.DataFrame:
    """Read a roster CSV (path or file-like) and index it by runner name."""
    df = pd.read_csv(source, dtype=str)
    return clean_roster(df)


def clean_roster(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "runner" not in df.columns:
        raise ValueError("Roster needs a 'runner' column")
    df["runner"] = df["runner"].astype(str).str.strip()
    df = df[df["runner"].ne("") & df["runner"].ne("nan")]
    if df["runner"].duplicated().any():
        dupes = sorted(df.loc[df["runner"].duplicated(), "runner"].unique())
        raise ValueError(f"Duplicate runner names: {', '.join(dupes)}")
    df = df.set_index("runner")
    return df.apply(lambda col: col.map(parse_time))


@dataclass
class CostMatrix:
    relay: Relay
    times: pd.DataFrame  # runners x legs, seconds (inf where the runner can't run the leg)
    sources: pd.DataFrame  # runners x legs: "split", "converted" or "missing"

    @property
    def runners(self) -> list[str]:
        return list(self.times.index)

    def values(self) -> np.ndarray:
        return self.times.to_numpy(dtype=float)


def build_cost_matrix(
    roster: pd.DataFrame,
    relay: Relay,
    adjustments: dict[str, Adjustment] | None = None,
    runners: list[str] | None = None,
) -> CostMatrix:
    """Expected time for every (runner, leg) pair of ``relay``.

    ``roster`` is indexed by runner (see :func:`clean_roster`). ``runners`` limits the
    matrix to the available runners, in that order.
    """
    adj = {**DEFAULT_ADJUSTMENTS, **(adjustments or {})}
    if runners is None:
        runners = list(roster.index)
    missing = [r for r in runners if r not in roster.index]
    if missing:
        raise ValueError(f"Not on the roster: {', '.join(missing)}")

    leg_names = [leg.name for leg in relay.legs]
    times = pd.DataFrame(math.inf, index=runners, columns=leg_names, dtype=float)
    sources = pd.DataFrame("missing", index=runners, columns=leg_names, dtype=object)

    for leg in relay.legs:
        split_col = relay.split_column(leg.number)
        open_col = open_column(leg.leg_type.open_event)
        for runner in runners:
            split = roster.at[runner, split_col] if split_col in roster.columns else math.nan
            if pd.notna(split):
                times.at[runner, leg.name] = float(split)
                sources.at[runner, leg.name] = "split"
                continue
            open_time = roster.at[runner, open_col] if open_col in roster.columns else math.nan
            if pd.notna(open_time):
                times.at[runner, leg.name] = adj[leg.leg_type.key].apply(float(open_time))
                sources.at[runner, leg.name] = "converted"

    times.index.name = sources.index.name = "runner"
    return CostMatrix(relay, times, sources)
