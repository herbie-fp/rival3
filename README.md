# Rival (paper name is Reval, they mean the same thing)

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
| `infra/*_plot.py`, `infra/histograms.py` | Report plots and printed evaluation statistics |
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

## Hardware requirements

The artifact was built and evaluated on a machine with an Intel Core i7-8700
CPU at 3.7 GHz, 32 GB of DDR4 RAM, and Ubuntu 24.04.2 LTS (Linux x86-64).

The evaluation requires no special hardware beyond a standard x86-64 machine
with 32 GB of RAM.

## Build the Docker image

Run these commands from the repository root. You need Docker and enough space
inside the container for the expanded benchmark input (about 460 MiB).
Building the image requires network access; running it does not.
The reference image was built with Docker 29.9.0 (build f415da8).

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
      python3 infra/repeats_plot.py -t /output/timeline.json
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
      python3 infra/repeats_plot.py -t /output/timeline.json
      cp infra/profile.js /output/profile.js
    } 2>&1 | tee /output/full-evaluation.log
  '
```

Once it's done, open `result/index.html` directly in a browser and check that
all five plots appear. The corresponding PNG and PDF files should also be in
`result/`. Some browsers block the profiling widget from loading local JSON
files; the plots remain available directly as PNG and PDF files.

Please, proceed to the "Verifying paper claims" once the full evaluation is done.

### Verifying paper claims

Please, compare obtained plots to the ones in the paper.
The mappings are:

| Plot | Paper figure |
| --- | --- |
| `ratio_plot_precision.pdf` | Figure 2 |
| `optimality_plot.pdf` | Figure 3 |
| `density_cdf_plot.pdf` | Figure 4 |
| `cnt_per_iters_plot.pdf` | Figure 5 |
| `histogram_valid.pdf` | Figure 6 |

Slight differences are expected for Figure 2, however, the trend should be the same: Rival should outperform other tools.

Almost no significant differences is expected for other figures.

#### RQ1

> All tools are run on 469 expressions totaling 567459 inputs.

Use `result/full-evaluation.log`, scroll to the bottom and find
```
NUMBER OF TUNED BENCHMARKS = ...
```
The number should be `469`.

Then, in the same log, find line 
```
\newcommand{\NumTunedPoints}{...\xspace}
```
The number should be `567459`.

> Reval is significantly faster, by 3.97× on average, and its advantage grows with difficulty: for the most difficult evaluations, which require precisions up to 2^14, Reval is 9.26× faster than Sollya.

> Reval is faster than Sollya at all precisions, 3.97× faster on average. Moreover its advantage increases with difficulty, up to 9.26× on the most difficult evaluations. The Ziv and Ziv+ ablations demonstrate that Reval’s advantage, especially on the most difficult evaluations, is due largely to Reval’s precision assignment algorithm.

Using `result/full-evaluation.log`, find

```
\newcommand{\RivalAvgSpeedupOverSollya}{...\xspace} 
```
It is an average speedup of Rival over Sollya, the exact number differ from run to run, but it should be around `4x` or higher

```
\newcommand{\RivalMaxSpeedupOverSollya}{...\xspace} 
```
It is a maximum speedup of Rival over Sollya, the exact number differ from run to run, but it should be around `10x` or higher

Using `result/ratio_plot_precision.pdf` make sure that Rival curve is above than Ziv and Ziv+ curves. The exact speedup might differ from run to run.

> Reval however, not only retains but grows its advantage, demonstrating that its precision assignment algorithm is largely responsible for its good performance on the most challenging inputs.

Using `result/ratio_plot_precision.pdf` make sure that Rival's curve is getting more above Ziv, Ziv+ and Sollya as precision increases (highest precisions might be noisy).

#### RQ2

> However, Reval’s precision assignment algorithm substantially outperforms both Ziv and Ziv+: 8.01× closer to the optimal precision assignment than Ziv+, and 14.41× closer than Ziv.

Using `result/full-evaluation.log`, find 
```
\newcommand{\RevalCloserToOptimalThanZiv}{...}
\newcommand{\RevalCloserToOptimalThanBaseline}{...}
```
Make sure that these numbers are `14.41` (closer than Ziv) and `8.01` (closer than Ziv+).

> Numerically, Reval executes 43.64% of primitives at 20% of the highest precision per all four tools, while Ziv+ executes 31.93% of primitives, and Ziv performs the worst with 1.06% of primitives. No tool reaches the optimal 50.41% of primitives executed at 20% of the highest prec

Using `result/full-evaluation.log`, find: 
```
\newcommand{\DensityPercentageOfLowerPrecisionReval}{...}
\newcommand{\DensityPercentageOfLowerPrecisionBaseline}{...}
\newcommand{\DensityPercentageOfLowerPrecisionZiv}{...}
\newcommand{\DensityPercentageOfLowerPrecisionOptimal}{...}
```
The numbers should be: `43.64`, `31.93`, `1.06`, `50.41`. Minor difference is unlikely but possible.

#### RQ3

> after two iterations, 59.05% of Reval’s inputs have converged while only 32.38% of Ziv+’s have, with Ziv performing about the same.

Using `result/full-evaluation.log`, find:
```
\newcommand{\RivalSecondIterConvergence}{...}
\newcommand{\BaselineSecondIterConvergence}{...}
```
These numbers should be: `59.05` and `32.38`, minor difference is possible, our latest run produced `59.07` and `32.39`.

> This leads to a radical reduction in the number of primitive computations needed by Reval: 31.7% fewer than Ziv+, 62.16% fewer than Ziv.

Using `result/full-evaluation.log`, find:
```
\newcommand{\RivalInstrCountLessThanBaseline}{...}
\newcommand{\RivalInstrCountLessThanZiv}{...}
```
The numbers should be `31.7` and `62.16`, minor difference is possible.

> The right most “tuning” column also shows that precision tuning itself is very fast — about 2.6% of total runtime.

Using `result/full-evaluation.log`, find:
```
\newcommand{\TuningTimePercentage}{...}
```
The number should be `2.6`, slight difference is possible, our latest run recorded `2.7`.

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

### Overview example

These commands evaluate `f(x) = (1 - cos(x)) / sin(x)` for different values of
`x`. Near zero, `f(x)` is approximately `x / 2`. Computing `1 - cos(x)`
subtracts two nearly equal numbers, so Rival may need more precision as `x`
gets smaller. The timings below will vary by machine.

For `x = 1e-8`:

```sh
docker run --rm --network none rival-artifact \
  /workspace/target/release/rival-cli \
  "(/ (- 1 (cos x)) (sin x))" "(x)" "(1e-8)"
```

```text
Executed 5 instructions for 2 iterations:

┌────────┬────────┬────────┬────────┬────────┐
│ Name   │ 0 Bits │ 0 Time │ 1 Bits │ 1 Time │
├────────┼────────┼────────┼────────┼────────┤
│ adjust │        │        │        │ 0.4 µs │
│ Cos    │     62 │ 1.6 µs │    117 │ 1.9 µs │
│ Sub    │     60 │ 0.3 µs │     60 │ 0.2 µs │
│ Sin    │     60 │ 2.4 µs │        │        │
│ Div    │     58 │ 0.3 µs │     58 │ 0.2 µs │
│ Total  │        │ 4.6 µs │        │ 2.8 µs │
└────────┴────────┴────────┴────────┴────────┘

Final value: [0.000000005, 0.000000005]
Total: 8.6 µs
```

For `x = 1e-18`:

```sh
docker run --rm --network none rival-artifact \
  /workspace/target/release/rival-cli \
  "(/ (- 1 (cos x)) (sin x))" "(x)" "(1e-18)"
```

```text
Executed 5 instructions for 2 iterations:

┌────────┬────────┬────────┬────────┬────────┐
│ Name   │ 0 Bits │ 0 Time │ 1 Bits │ 1 Time │
├────────┼────────┼────────┼────────┼────────┤
│ adjust │        │        │        │ 0.5 µs │
│ Cos    │     62 │ 0.3 µs │    178 │ 2.4 µs │
│ Sub    │     60 │ 0.3 µs │     60 │ 0.2 µs │
│ Sin    │     60 │ 0.2 µs │        │        │
│ Div    │     58 │ 0.3 µs │     58 │ 0.3 µs │
│ Total  │        │ 1.1 µs │        │ 3.4 µs │
└────────┴────────┴────────┴────────┴────────┘

Final value: [0.0000000000000000005, 0.0000000000000000005]
Total: 5.7 µs
```

For `x = 1e-26`:

```sh
docker run --rm --network none rival-artifact \
  /workspace/target/release/rival-cli \
  "(/ (- 1 (cos x)) (sin x))" "(x)" "(1e-26)"
```

```text
Executed 5 instructions for 3 iterations:

┌────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│ Name   │ 0 Bits │ 0 Time │ 1 Bits │ 1 Time │ 2 Bits │ 2 Time │
├────────┼────────┼────────┼────────┼────────┼────────┼────────┤
│ adjust │        │        │        │ 0.4 µs │        │ 0.5 µs │
│ Cos    │     62 │ 0.3 µs │    178 │ 2.6 µs │    236 │ 2.8 µs │
│ Sub    │     60 │ 0.3 µs │     60 │ 0.3 µs │     60 │ 0.3 µs │
│ Sin    │     60 │ 0.2 µs │        │        │        │        │
│ Div    │     58 │ 0.4 µs │     58 │ 0.2 µs │     58 │ 0.3 µs │
│ Total  │        │ 1.2 µs │        │ 3.5 µs │        │ 4.0 µs │
└────────┴────────┴────────┴────────┴────────┴────────┴────────┘

Final value: [0.000000000000000000000000005, 0.000000000000000000000000005]
Total: 10.6 µs
```

For `x = 1e-80`:

```sh
docker run --rm --network none rival-artifact \
  /workspace/target/release/rival-cli \
  "(/ (- 1 (cos x)) (sin x))" "(x)" "(1e-80)"
```

```text
Executed 5 instructions for 4 iterations:

┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│ Name   │ 0 Bits │ 0 Time │ 1 Bits │ 1 Time │ 2 Bits │ 2 Time │ 3 Bits │ 3 Time │
├────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
│ adjust │        │        │        │ 0.5 µs │        │ 0.4 µs │        │ 0.3 µs │
│ Cos    │     62 │ 0.3 µs │    178 │ 0.3 µs │    347 │ 0.3 µs │    622 │ 6.0 µs │
│ Sub    │     60 │ 0.3 µs │     60 │ 0.2 µs │     60 │ 0.3 µs │     60 │ 0.3 µs │
│ Sin    │     60 │ 0.2 µs │        │        │        │        │        │        │
│ Div    │     58 │ 0.3 µs │     58 │ 0.2 µs │     58 │ 0.2 µs │     58 │ 0.3 µs │
│ Total  │        │ 1.2 µs │        │ 1.2 µs │        │ 1.1 µs │        │ 6.9 µs │
└────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘

Total: 12.5 µs
```

The final decimal interval is very long; both endpoints are approximately
`5e-81`.

### Case studies

#### Slack mechanism

This example evaluates `cos(x) - cos(x + eps)` with a very large `x` and a
very small `eps` causing a catastrophic cancellation.

```sh
docker run --rm --network none rival-artifact \
  /workspace/target/release/rival-cli \
  "(- (cos x) (cos (+ x eps)))" "(x eps)" "(1e100 1e-100)"
```

```text
Executed 4 instructions for 4 iterations:

┌────────┬────────┬────────┬────────┬─────────┬────────┬─────────┬────────┬─────────┐
│ Name   │ 0 Bits │ 0 Time │ 1 Bits │ 1 Time  │ 2 Bits │ 2 Time  │ 3 Bits │ 3 Time  │
├────────┼────────┼────────┼────────┼─────────┼────────┼─────────┼────────┼─────────┤
│ adjust │        │        │        │  0.7 µs │        │  0.4 µs │        │  0.4 µs │
│ Cos    │     60 │ 5.9 µs │    117 │  6.2 µs │    284 │  9.2 µs │    556 │ 13.7 µs │
│ Add    │     62 │ 0.4 µs │    507 │  0.3 µs │    620 │  0.4 µs │    892 │  0.4 µs │
│ Cos    │     60 │ 0.2 µs │    118 │  7.1 µs │    284 │  9.9 µs │    556 │ 15.0 µs │
│ Sub    │     58 │ 0.4 µs │     58 │  0.3 µs │     58 │  0.4 µs │     58 │  0.3 µs │
│ Total  │        │ 6.9 µs │        │ 14.6 µs │        │ 20.3 µs │        │ 29.8 µs │
└────────┴────────┴────────┴────────┴─────────┴────────┴─────────┴────────┴─────────┘

Final value: [-0.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000003806377310050287, -0.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000003806377310050287]
Total: 74.1 µs
```

#### Correct rounding

This example evaluates `(x + y) × (z + 1)` with inputs that cause double rounding.

```sh
docker run --rm --network none rival-artifact \
  /workspace/target/release/rival-cli \
  "(* (+ x y) (+ z 1))" "(x y z)" \
  "(1.3002052657264033e189  3.084776002356433e188 9.332636e-302)"
```

```text
Executed 4 instructions for 7 iterations:

┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│ Name   │ 0 Bits │ 0 Time │ 1 Bits │ 1 Time │ 2 Bits │ 2 Time │ 3 Bits │ 3 Time │ 4 Bits │ 4 Time │ 5 Bits │ 5 Time │ 6 Bits │ 6 Time │
├────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
│ adjust │        │        │        │ 0.3 µs │        │ 0.2 µs │        │ 0.2 µs │        │ 0.2 µs │        │ 0.2 µs │        │ 0.2 µs │
│ Add    │     60 │ 0.3 µs │        │        │        │        │        │        │        │        │        │        │        │        │
│ Add    │     60 │ 0.4 µs │    113 │ 0.3 µs │    166 │ 0.2 µs │    272 │ 0.2 µs │    484 │ 0.2 µs │    908 │ 0.3 µs │   1756 │ 0.3 µs │
│ Mul    │     58 │ 0.3 µs │    111 │ 0.3 µs │    164 │ 0.3 µs │    270 │ 0.4 µs │    482 │ 0.4 µs │    906 │ 0.4 µs │   1754 │ 0.4 µs │
│ Total  │        │ 1.0 µs │        │ 0.9 µs │        │ 0.8 µs │        │ 0.8 µs │        │ 0.8 µs │        │ 0.9 µs │        │ 1.0 µs │
└────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘

Final value: 1608682865962046700000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
Total: 10.1 µs
```
