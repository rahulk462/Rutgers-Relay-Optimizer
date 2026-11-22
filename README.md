# Rutgers Relay Optimizer

Enter each runner's times and get the fastest relay lineup: who runs, and on which leg.
Coaches usually make this call by feel; this tool does it in seconds and shows how much time
each alternative costs.

Version 1 handles one relay at a time (4x100, 4x400 or DMR) with coach-entered times, and returns:

- the best lineup, leg by leg, with its predicted total time
- the next fastest alternatives and how many seconds slower each is
- how the coach's preferred lineup compares ("your lineup is 0.18 s slower")
- how much each runner's time would have to change to flip the order

Locks ("Smith anchors") and exclusions ("Jones out") are supported.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Run the tests with `pytest`.

To deploy for free, push this repo to GitHub and create an app on
[Streamlit Community Cloud](https://streamlit.io/cloud) pointing at `app.py`.

## Roster format

A CSV with a `runner` column plus any of these (blank means unknown). Times may be seconds
(`10.85`) or minutes (`4:05.3`).

| Column | Meaning |
| --- | --- |
| `open_100`, `open_400`, `open_800`, `open_1200`, `open_1600` | Open (individual race) times |
| `<relay>_leg<n>`, e.g. `4x100_leg3`, `4x400_leg1`, `DMR_leg2` | A known split for that exact leg; used as is |

See [`sample_data.csv`](sample_data.csv) for a made-up roster.

## How it works

It's an assignment problem. Let c(i, j) be runner i's expected time on leg j, and x(i, j) = 1
if runner i runs leg j. Minimize the sum of c(i, j) × x(i, j) such that every leg gets exactly
one runner and every runner runs at most one leg.

**Building c** (`times.py`): a known split for the leg is used directly. Otherwise the runner's
open time for that leg's event is converted with an adjustment for the leg type:
`expected = open × factor + offset`. Legs differ, so each kind gets its own adjustment:

| Relay | Leg | Leg type | Default offset |
| --- | --- | --- | --- |
| 4x100 | 1 | blocks, curve | +0.10 s |
| 4x100 | 2, 4 | flying, straight | −0.90 s |
| 4x100 | 3 | flying, curve | −0.75 s |
| 4x400 | 1 | blocks, all in lanes | +0.20 s |
| 4x400 | 2 | flying, cuts in | −0.70 s |
| 4x400 | 3, 4 | flying, in traffic | −0.50 s |
| DMR | 1 (1200 m) | standing start | 0 |
| DMR | 2 (400 m) | flying | −0.70 s |
| DMR | 3 (800 m) | flying | −0.50 s |
| DMR | 4 (1600 m) | flying | −0.50 s |

These defaults are rough starting guesses; the coach can edit every factor and offset in the app.

**Solving** (`solver.py`): brute force checks every ordered lineup (8 runners give
8 × 7 × 6 × 5 = 1,680; 10 give 5,040), which is instant and gives the full ranking.
`scipy.optimize.linear_sum_assignment` (the Hungarian algorithm) solves for the best lineup
directly, and every solve checks that the two agree. Locks and exclusions are applied as
infinite costs.

## Code layout

| File | What it does |
| --- | --- |
| `model.py` | Relay definitions: legs, leg types, distances |
| `times.py` | Time parsing, roster loading, and the cost matrix from splits or converted open times |
| `solver.py` | Brute force, Hungarian, top-k ranking, preferred-lineup comparison, sensitivity |
| `app.py` | The Streamlit page |
| `tests/` | Solver agreement, constraints, conversions, and an app smoke test |
| `sample_data.csv` | A made-up roster for the demo |

## Later

- Solving several relays at once with no runner in two overlapping relays, plus per-meet event
  limits (an integer program with PuLP or OR-Tools).
- Baton exchange quality for specific pairs of runners.
