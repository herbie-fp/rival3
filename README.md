# Rival 3

Rival evaluates real expressions to high precision. This repository also
contains the timing evaluation used to compare Rival with the baseline, Ziv,
and Sollya evaluators. Evaluators can build and run it with Docker on Linux
amd64; the image includes the benchmark data and required tools.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/` | Rival's Rust library and `rival-cli` |
| `rival3-ffi/` | Native library used by the Racket bindings |
| `rival3-racket/` | Racket package exposing Rival |
| `infra/points.json.xz` | Compressed timing benchmark input |
| `infra/time.rkt` | Benchmark runner and HTML/timeline generator |
| `infra/*_plot.py`, `infra/histograms.py` | Plot generators used by the HTML report |
| `Dockerfile` | Pinned Linux amd64 evaluation environment |

### Rust source map

| Path | Purpose |
| --- | --- |
| `src/lib.rs` | Public library API and usage examples |
| `src/bin/main.rs` | `rival-cli` argument parsing and output |
| `src/mpfr.rs` | Low-level MPFR wrappers |
| `src/eval/mod.rs` | Evaluator module wiring |
| `src/eval/machine.rs` | Expression compilation and register-machine setup |
| `src/eval/run.rs` | Main evaluation loop and convergence checks |
| `src/eval/optimal.rs` | Per-point optimal precision search |
| `src/eval/profile.rs` | Per-instruction execution profiles |
| `src/eval/instructions.rs` | Register-machine instruction types |
| `src/eval/execute.rs` | Execution of individual instructions |
| `src/eval/ops.rs` | Expression operations and optimization rules |
| `src/eval/macros.rs` | Macros that generate operation dispatch |
| `src/eval/tricks.rs` | Precision-bound calculations |
| `src/eval/adjust/mod.rs` | Adaptive precision adjustment and path reduction |
| `src/eval/adjust/precision.rs` | Per-instruction precision tuning |
| `src/eval/adjust/path_reduction.rs` | Skipping or simplifying unneeded computation paths |
| `src/interval/mod.rs` | Interval module wiring and exports |
| `src/interval/value.rs` | Interval values, endpoints, and error flags |
| `src/interval/core.rs` | Shared interval operations and helpers |
| `src/interval/arithmetic.rs` | Basic arithmetic and related operations |
| `src/interval/boolean.rs` | Boolean intervals, comparisons, and conditionals |
| `src/interval/constants.rs` | Mathematical constants |
| `src/interval/fmod.rs` | Remainder operations |
| `src/interval/gamma.rs` | Gamma and log-gamma operations |
| `src/interval/pow.rs` | Power operations |
| `src/interval/trig.rs` | Trigonometric operations |

Run the following commands from the repository root. You need Docker and
enough free space to expand the benchmark input (about 460 MiB) inside the
container. Building the image requires network access; running it does not.

```sh
docker build --platform linux/amd64 -t rival3-timing:local .
mkdir -p result
```

`result` is a directory on your host. Both commands below mount it at
`/output` inside the container, so the generated files remain in `result/`
after the container exits.

## Kick the tires

```sh
docker run --rm --network none --mount type=bind,src="$PWD/result",dst=/output \
  rival3-timing:local bash -lc '
    set -euo pipefail
    {
      xz -dc infra/points.json.xz > /tmp/points.json
      racket infra/smoke-test.rkt
      racket infra/time.rkt --id 4 /tmp/points.json
    } 2>&1 | tee /output/kick-the-tires.log
  '
```

This runs the Racket binding smoke test and then benchmark record `4` (the
fifth record) against every point in the real dataset. It exercises Rival,
Sollya, and the optimal precision calculation. Check
`result/kick-the-tires.log` for the selected expression and the line beginning
`4:`. This short run does not generate the HTML report or plots.

## Full evaluation

Run this on native Linux amd64. It evaluates every benchmark record and then
generates the report and plots; it can take substantially longer than the
kick-the-tires run.

```sh
docker run --rm --network none --mount type=bind,src="$PWD/result",dst=/output \
  rival3-timing:local bash -lc '
    set -euo pipefail
    {
      xz -dc infra/points.json.xz > /tmp/points.json
      racket infra/time.rkt --dir /output --profile /output/profile.json /tmp/points.json
      python3 infra/ratio_plot.py -t /output/timeline.json -o /output
      python3 infra/histograms.py -t /output/timeline.json -o /output
      python3 infra/cnt_per_iters_plot.py -t /output/timeline.json -o /output
      python3 infra/density_plot.py -t /output/timeline.json -o /output
      python3 infra/optimality_plot.py -t /output/timeline.json -o /output
      cp infra/profile.js /output/profile.js
    } 2>&1 | tee /output/full-evaluation.log
  '
```

The results are on the host in `result/`:

| File | Contents |
| --- | --- |
| `index.html` | Timing table and plot links |
| `timeline.json` | Data used to make the plots |
| `profile.json` | Racket profiling data |
| `*.png` | The five plots referenced by the HTML report |
| `full-evaluation.log` | Progress and summary for every record |

Open `result/index.html` directly in a browser for the table and plots.
`profile.json` is available alongside it. Some browsers block the HTML
profiling widget from loading local JSON files. To check completion, confirm
`result/full-evaluation.log` contains the `Total Time` and `Total Memory`
summary and that `result/index.html`, `result/timeline.json`,
`result/profile.json`, and the referenced PNG files exist.

The image includes `infra/points.json.xz` and expands it in temporary container
storage for each run. It does not require the separate optimal precision cache.
Dependency versions are pinned in the [Dockerfile](Dockerfile).

## Try the Rust CLI

With Rust 1.85 or newer installed, you can try Rival directly from the
repository root. Pass an expression, its variables, and their input values:

```sh
cargo run --bin rival-cli --release -- "(- (sqrt (+ x 1)) 1)" "(x)" "(1e-30)"
```

You should see a per-instruction table like this (timings vary by machine):

```text
Executed 4 instructions for 2 iterations:

┌────────┬────────┬────────┬────────┬────────┐
│ Name   │ 0 Bits │ 0 Time │ 1 Bits │ 1 Time │
├────────┼────────┼────────┼────────┼────────┤
│ adjust │        │        │        │ 0.5 µs │
│ Add    │     62 │ 0.3 µs │    633 │ 0.2 µs │
│ Sqrt   │     60 │ 0.8 µs │    632 │ 1.6 µs │
│ Sub    │     58 │ 0.2 µs │     58 │ 0.2 µs │
│ Total  │        │ 1.2 µs │        │ 2.6 µs │
└────────┴────────┴────────┴────────┴────────┘

Final value: [0.0000000000000000000000000000005, 0.0000000000000000000000000000005]
Total: 5.2 µs
```

The `Bits` columns show the precision used for each instruction in each
iteration. The final value is the computed interval.

