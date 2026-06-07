# nined_energy_memory

`nined_energy_memory` is the CLBench adapter for the 9D bounded online memory
runtime. The registered system package is benchmark plumbing; the tested object
is bounded online state updated from streaming task evidence.

Implemented BSM modes:

```text
bsm_energy
bsm_vanilla_online
bsm_static_pipeline
bsm_current_scan_only
bsm_no_write
bsm_no_propagation
bsm_sliding_window
```

for `blind_spectrum_monitoring`.

The BSM adapter parses visible spectrum peaks, updates a fixed-capacity energy
landscape over candidate transmitter basins, and decodes low-energy basins into
the task's `ScanReport` schema.

Attribution result as of 2026-06-06:

```text
bsm_vanilla_online > bsm_energy > archived GPT-5.4 ICL > bsm_sliding_window
>> bsm_static_pipeline ~= bsm_current_scan_only ~= bsm_no_write
```

So the BSM result supports bounded full-horizon online accumulation, not an
energy-specific mechanism claim. Keep `bsm_energy` as a mechanism comparison;
use `bsm_vanilla_online` as the current best BSM product/runtime arm.

Implemented Database Exploration modes:

```text
db_static_policy
db_current_question_only
db_no_write
db_sliding_window
db_hard_cache
db_vanilla_online
db_energy
```

for `database_exploration`.

The Database adapter uses a shared deterministic SQL planner across all arms.
The planner first discovers public schema through the task `QUERY` interface,
then asks answer SQL queries and submits the observed scalar result. Persistent
database memory stores schema facts, table/column basins, drift events, energy
decompositions, and MM^T-style landscape diagnostics. The runtime never opens
SQLite directly, never reads the question files, and never receives ground-truth
SQL before public feedback.

Database claim boundary:

```text
db_energy > db_vanilla_online and db_energy > db_hard_cache
```

is required for an energy-specific mechanism claim.

```text
db_vanilla_online or db_hard_cache > db_no_write
```

supports only a bounded-online-runtime claim.

If `db_static_policy`, `db_current_question_only`, or `db_no_write` match the
best memory arm, the result is a SQL-planner result rather than a memory result.

Implemented Sales Prediction modes:

```text
sales_static_policy
sales_current_instance_only
sales_no_write
sales_sliding_window
sales_hard_cache
sales_vanilla_online
sales_energy
```

for `sales_prediction`.

The Sales adapter interacts through the public bash interface.  It runs one
inspection command that reads `/app/data/*.csv`, `furniture.json`,
`furniture_types.json`, and `locations.json`, then submits structured
`PredictionResponse` entries.  Persistent modes accumulate public sales rows and
public previous-round feedback; static/current-only arms isolate whether the
score comes from the forecaster itself or from retained stream evidence.

Sales claim boundary:

```text
sales_energy > sales_vanilla_online and sales_energy > sales_hard_cache
```

is required for an energy-specific mechanism claim.

```text
sales_vanilla_online, sales_hard_cache, or sales_sliding_window > sales_no_write
```

supports only a bounded-online-runtime claim.
