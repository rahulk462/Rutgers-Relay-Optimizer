"""Streamlit page for the relay lineup optimizer. Run with ``streamlit run app.py``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from model import RELAYS, get_relay, open_column
from solver import InfeasibleError, evaluate, optimize, rank_of, sensitivity
from times import DEFAULT_ADJUSTMENTS, Adjustment, build_cost_matrix, clean_roster, format_time

SAMPLE = Path(__file__).with_name("sample_data.csv")
ANY = "(any)"
NONE = "(none)"

st.set_page_config(page_title="Relay Lineup Optimizer", page_icon="🏃", layout="wide")
st.title("Relay lineup optimizer")
st.caption(
    "Enter each runner's times and get the fastest lineup: who runs, and on which leg, "
    "plus how much time each alternative costs."
)

# ---------------------------------------------------------------- inputs
with st.sidebar:
    relay_key = st.selectbox("Relay", list(RELAYS), format_func=lambda k: RELAYS[k].name)
    relay = get_relay(relay_key)
    upload = st.file_uploader("Roster CSV", type="csv", help="Leave empty to use the sample roster.")
    n_alternatives = st.slider("Alternatives to show", 1, 15, 5)

raw = pd.read_csv(upload if upload is not None else SAMPLE, dtype=str).fillna("")
if upload is None:
    st.info("Using the made-up sample roster. Upload your own CSV in the sidebar, or edit the table below.")

relevant_cols = ["runner"]
for leg in relay.legs:
    for col in (open_column(leg.leg_type.open_event), relay.split_column(leg.number)):
        if col not in relevant_cols:
            relevant_cols.append(col)
for col in relevant_cols:
    if col not in raw.columns:
        raw[col] = ""

st.subheader("1. Runner times")
st.write(
    "Seconds (`10.85`) or minutes (`4:05.3`). A split for a specific leg (e.g. `4x100_leg3`) "
    "is used as is; otherwise the open time is converted with the leg adjustments below. "
    "Add rows for more runners."
)
edited = st.data_editor(
    raw[relevant_cols],
    num_rows="dynamic",
    width="stretch",
    hide_index=True,
    key=f"roster_{relay_key}_{getattr(upload, 'name', 'sample')}",
)

try:
    roster = clean_roster(edited)
except ValueError as err:
    st.error(str(err))
    st.stop()

st.subheader("2. Leg adjustments")
st.write("Expected leg time = open time × factor + offset (seconds). These are starting guesses; edit them.")
leg_types = list(dict.fromkeys(leg.leg_type for leg in relay.legs))
adj_df = pd.DataFrame(
    {
        "leg type": [lt.label for lt in leg_types],
        "from open": [f"{lt.open_event} m" for lt in leg_types],
        "factor": [DEFAULT_ADJUSTMENTS[lt.key].factor for lt in leg_types],
        "offset (s)": [DEFAULT_ADJUSTMENTS[lt.key].offset for lt in leg_types],
    }
)
adj_edit = st.data_editor(
    adj_df,
    hide_index=True,
    disabled=["leg type", "from open"],
    column_config={
        "factor": st.column_config.NumberColumn(format="%.3f", step=0.001),
        "offset (s)": st.column_config.NumberColumn(format="%.2f", step=0.05),
    },
    key=f"adj_{relay_key}",
)
adjustments = {
    lt.key: Adjustment(float(row["factor"]), float(row["offset (s)"]))
    for lt, (_, row) in zip(leg_types, adj_edit.iterrows())
}

cost_all = build_cost_matrix(roster, relay, adjustments)
pool = [r for r in cost_all.runners if (cost_all.sources.loc[r] != "missing").any()]

st.subheader("3. Who's available")
c1, c2 = st.columns(2)
with c1:
    out = st.multiselect("Out", pool, help="Runners who can't run this relay.")
available = [r for r in pool if r not in out]
with c2:
    st.write(f"**{len(available)} available:** " + (", ".join(available) or "none"))

leg_cols = st.columns(relay.n_legs)
locks: dict[int, str] = {}
preferred: list[str] = []
for j, (leg, col) in enumerate(zip(relay.legs, leg_cols)):
    with col:
        st.markdown(f"**{leg.name}** · {leg.distance_m} m")
        st.caption(leg.leg_type.description)
        lock = st.selectbox("Lock", [ANY, *available], key=f"lock_{relay_key}_{j}")
        if lock != ANY:
            locks[j] = lock
        pref = st.selectbox("Your lineup", [NONE, *available], key=f"pref_{relay_key}_{j}")
        preferred.append(pref)

# ---------------------------------------------------------------- solve
cost = build_cost_matrix(roster, relay, adjustments, runners=available)
try:
    result = optimize(cost, k=n_alternatives + 1, locks=locks)
except InfeasibleError as err:
    st.error(f"No lineup possible: {err}")
    st.stop()

st.divider()
best = result.best
st.subheader(f"Fastest lineup: {format_time(best.total)}")
st.dataframe(
    pd.DataFrame(
        {
            "leg": [leg.name for leg in relay.legs],
            "runner": best.runners,
            "predicted": [format_time(t) for t in best.leg_times],
            "based on": [cost.sources.at[r, leg.name] for r, leg in zip(best.runners, relay.legs)],
        }
    ),
    hide_index=True,
    width="stretch",
)

if any(p != NONE for p in preferred):
    if NONE in preferred:
        st.warning("Pick a runner for every leg to compare your lineup.")
    elif len(set(preferred)) < len(preferred):
        st.warning("Your lineup uses the same runner twice.")
    else:
        mine = evaluate(cost, preferred)
        gap = mine.total - best.total
        if gap <= 1e-9:
            st.success(f"Your lineup ({format_time(mine.total)}) is the fastest.")
        else:
            place = rank_of(mine, result.all_lineups) if all(
                mine.runners[j] == r for j, r in locks.items()
            ) else None
            where = f", #{place} of {result.n_feasible} possible lineups" if place else ""
            st.warning(f"Your lineup ({format_time(mine.total)}) is **{gap:.2f} s slower**{where}.")

st.subheader("Next fastest alternatives")
if result.alternatives:
    rows = []
    for n, alt in enumerate(result.alternatives, start=2):
        row = {"#": n}
        row.update({leg.name: r for leg, r in zip(relay.legs, alt.runners)})
        row["total"] = format_time(alt.total)
        row["slower by"] = f"+{alt.total - best.total:.2f} s"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
else:
    st.write("This is the only possible lineup.")

st.subheader("What would change the order")
sens = sensitivity(result, cost)
if sens:
    lines = []
    for s in sens:
        leg_name = relay.legs[s.leg].name
        if s.in_lineup:
            lines.append(f"- **{s.runner}** ({leg_name}) can be up to **{s.seconds:.2f} s** slower before the best lineup changes.")
        else:
            lines.append(f"- **{s.runner}** would need to be **{s.seconds:.2f} s** faster on {leg_name} to make the lineup.")
    st.markdown("\n".join(lines))

with st.expander("Expected time for every runner on every leg"):
    shown = cost.times.map(format_time)
    st.dataframe(shown, width="stretch")
    st.caption("— means no time for that leg. Splits are used as entered; everything else is converted.")
