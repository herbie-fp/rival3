# Rival 3

Rival evaluates real expressions to high precision. This repository contains a
timing evaluation comparing Rival with the Ziv+, Ziv, and Sollya evaluators.
The artifact runs with Docker on Linux amd64; its image includes the benchmark
data and required tools.

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

## Build the Docker image

Run these commands from the repository root. You need Docker and enough space
inside the container for the expanded benchmark input (about 460 MiB).
Building the image requires network access; running it does not.

```sh
docker build --platform linux/amd64 -t rival-artifact .
mkdir -p result
```

The image uses these versions on Linux amd64:

| Component | Version |
| --- | --- |
| Ubuntu | 24.04 (base-image digest pinned) |
| Rust | 1.88.0 |
| Racket | 8.18 |
| Sollya | 8.0 |
| GMP | 6.3.0 |
| MPFR | 4.2.1 |
| MPFI | 1.5.3 |
| `gmp-mpfr-sys` | 1.6.4 |
| Python plotting libraries | NumPy 1.26.4, pandas 2.1.4, Matplotlib 3.6.3 |

Evaluation commands mount `result/` directory at
`/output` inside the container, so generated files remain in `result/` after
the container exits.

Either evaluation writes these files to `result/`:

| File | Contents |
| --- | --- |
| `index.html` | Timing table and plots |
| `timeline.json` | Data used to make the plots |
| `profile.json` | Racket profiling data |
| `*.png`, `*.pdf` | The five report plots, each in both formats |
| `kick-the-tires.log` or `full-evaluation.log` | Progress and summary for the selected run |


## Kick the tires (a few minutes)

This runs one benchmark record (number 4) against all its points. It evaluates Rival,
Ziv+, Ziv, and Sollya, and calculates optimal precision. The report and plots
use only this record, so they will differ from the full evaluation.

```sh
rm -f -- result/*  # Clean up result directory
docker run --rm --network none --mount type=bind,src="$PWD/result",dst=/output \
  rival-artifact bash -lc '
    set -euo pipefail
    {
      xz -dc infra/points.json.xz > /tmp/points.json
      racket infra/time.rkt --id 4 --dir /output --profile /output/profile.json /tmp/points.json
      python3 infra/ratio_plot.py -t /output/timeline.json -o /output
      python3 infra/histograms.py -t /output/timeline.json -o /output
      python3 infra/cnt_per_iters_plot.py -t /output/timeline.json -o /output
      python3 infra/density_plot.py -t /output/timeline.json -o /output
      python3 infra/optimality_plot.py -t /output/timeline.json -o /output
      cp infra/profile.js /output/profile.js
    } 2>&1 | tee /output/kick-the-tires.log
  '
```

Once it's done, open `result/index.html` directly in a browser and check that
all five plots appear. The corresponding PNG and PDF files should also be in
`result/`. Some browsers block the profiling widget from loading local JSON
files; the plots remain available directly as PNG and PDF files.

## Full evaluation (may take hours)

Run this on native Linux amd64. It processes every benchmark record, evaluates
Rival, Ziv+, Ziv, and Sollya, calculates optimal precision, and generates the
report and plots in `result/`.

```sh
rm -f -- result/*  # Clean up result directory
docker run --rm --network none --mount type=bind,src="$PWD/result",dst=/output \
  rival-artifact bash -lc '
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

Once it's done, open `result/index.html` directly in a browser and check that
all five plots appear. The corresponding PNG and PDF files should also be in
`result/`. Some browsers block the profiling widget from loading local JSON
files; the plots remain available directly as PNG and PDF files.

## Try the Rust CLI

After building the Docker image above, you can try Rival's CLI without
installing Rust or its dependencies on your host. Pass an expression, its
variables, and their input values:

```sh
docker run --rm --network none rival-artifact \
  /workspace/target/release/rival-cli \
  "(- (sqrt (+ x 1)) 1)" "(x)" "(1e-30)"
```

You should see a per-instruction table like this (timings and intermediate
precisions may vary):

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

