# rust-rs
Rival is a library for evaluating real expressions to high precision. Port of https://github.com/herbie-fp/rival. You can play around with the CLI:

```
$ cargo run --bin rival-cli --release -- "(- (sqrt (+ x 1)) 1)" "(x)" "(1e-30)"
```
Will give you:
```
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

## Port Notes
* File breakdown:
  * `eval/`
    * `adjust/`: Backwards pass logic
    * `execute.rs`: Evaluate an instruction at a precision (should consider moving to `macros.rs` or moving out a lot of the macro generated functions into similar files)
    * `instructions.rs`: The machine instructions
    * `machine.rs`: Rival machine builder and state; lowers expressions and initializes the machine
    * `macros.rs`: Generates the entire AST, instruction specification, amplification bounds, optimizations, path reductions, and more; provides the `def_ops!` macro
    * `ops.rs`: The operation registry using `def_ops!` (more on this below)
    * `run.rs`: Public Rival API
    * `tricks.rs`: More or less helper functions that are used when specifying amplification bounds in `ops.rs`
  * `interval/`
    * `value.rs`: Defines `Ival`, `Endpoint`, and `ErrorFlags`
    * Everything else is the same as `ops/*.rkt` in Racket Rival
  * `mpfr.rs`: Unsafe wrappers around MPFR
* Appears to be much faster than Racket Rival:
  * 3-5x faster for complex, MPFR dominant computations (I suspect Racket has some heavy MPFR FFI costs)
  * \>30x faster for smaller, Racket runtime dominant computations
* A number of TODO comments are spread throughout the code (mostly about efficiency)
* I ran both the Racket Rival and Rust Rival on a large portion of `points.json` (from Racket Rival's `infra/`) and compared machine state per iteration. Observations:
  * Compared per iteration precisions, repeats, etc.
  * Both Racket Rival and Rust Rival always output a correct final value
  * Final precisions are almost always the same, and in the cases where they differ, I believe it's a bug in the printing logic rather than a difference in the code
  * Rust Rival was noticeably faster
* What's missing:
  * Early stopping when we exceed max precision
  * Proper handling of rationals
    * Currently both the numerator and denominator are expressed as `u64`
    * Overflows when run on certain expressions in `jmatjs.fpcore` from the Herbie benchmark
    * Workaround for now is to just convert rationals to intervals
  * `ops/gamma.rkt` is not ported
  * Proper interval arithmetic tests
  * Nightly benchmarks + Sollya
* All operations are mutating
  * Every operation looks like `out.mul_assign(lhs, rhs)` where `out`, `lhs`, and `rhs` are all `Ival`s
* Registering operations across `eval/` can be done simply using a macro! This macro (in `eval/ops.rs`) will generate the entire AST, instruction specification, amplification bounds, optimizations, path reductions, and more:
  ```rust
  def_ops! {
    constant { ... }
    unary {
        Exp: {
            method: exp_assign,
            bounds: |ctx, out, inp| {
                let upper = ctx.maxlog(inp, false) + ctx.logspan(out);
                let lower = if ctx.lower_bound_early_stopping {
                    ctx.minlog(inp, true)
                } else { 0 };
                AmplBounds::new(upper, lower)
            },
            optimize: |arg| {
                // Exp(Log(x)) => x
                if let Log(x) = arg { *x } else { Exp(Box::new(arg)) }
            },
        },
        Not: {
            method: not_assign,
            bounds: |_, _, _| AmplBounds::zero(),
            path_reduce: path_reduction::not_op_path_reduce,
        },
    }
    ...
    unary_param { ... }  // sinu, cosu, tanu
    binary { ... }
    ternary { ... }
  }
  ```
  Thus, adding a new operation simply requires a user to create an `Ival` method (in `interval/`) to perform the interval arithmetic, and then register it in `ops.rs`.