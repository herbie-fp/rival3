//! Main evaluation loop with adaptive precision tuning.

use std::time::Instant;

use itertools::{enumerate, izip};
use thiserror::Error;

use crate::eval::{
    execute,
    input::{Input, InputRange},
    machine::{Discretization, Hint, Machine, Strategy},
    profile::Execution,
};
use crate::interval::Ival;

/// Selects when an invalid output makes an evaluation invalid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputPolicy {
    /// Any totally invalid output makes the whole evaluation invalid.
    RequireAll,
    /// Only a totally invalid *set* of outputs makes the evaluation invalid.
    /// Individual invalid outputs are returned in place.
    AllowPartial,
}

impl<D: Discretization> Machine<D> {
    /// Evaluate the compiled real expressions on an input point
    /// represented as a slice of typed inputs.
    ///
    /// Input conversion instructions call [`Ival::with_val`] at the precision
    /// requested by the evaluation strategy, retaining original values for
    /// refinement. [`Input::from`] accepts strings, booleans, primitive numbers,
    /// Rug numeric types, and existing intervals. Inputs may change type between
    /// calls. All declared arguments must be supplied, including unused ones.
    ///
    /// `args` must be the same length as the `vars` passed to
    /// [`MachineBuilder::build`](super::machine::MachineBuilder::build). The output is a vector of output
    /// values of the same length as the `exprs` passed to
    /// [`MachineBuilder::build`](super::machine::MachineBuilder::build).
    ///
    /// `hint` can be provided from a previous call to
    /// [`Machine::analyze_with_hints`] to speed up evaluation.
    /// Pass `None` for default behavior.
    ///
    /// `max_iterations` sets the maximum number of re-evaluation
    /// iterations before giving up. A [`Strategy::Baseline`] machine
    /// ignores it and gives up once the precision would exceed
    /// [`Machine::max_precision`].
    ///
    /// # Errors
    ///
    /// Returns [`RivalError::InvalidInput`] according to the selected
    /// [`OutputPolicy`].
    /// Returns [`RivalError::Unsamplable`] if Rival is unable to
    /// evaluate at least one expression.
    /// Returns [`RivalError::InputParse`] for malformed strings encountered
    /// during evaluation, with the argument index.
    ///
    /// Under [`OutputPolicy::AllowPartial`], totally invalid outputs are
    /// returned in place, while every non-error output must still be
    /// correctly rounded.
    pub fn apply(
        &mut self,
        args: &[Input],
        hint: Option<&[Hint]>,
        max_iterations: usize,
        policy: OutputPolicy,
    ) -> Result<Vec<Ival>, RivalError> {
        self.reset_evaluation(args.len());
        let hint_storage;
        let hint_slice: &[Hint] = if let Some(h) = hint {
            h
        } else {
            hint_storage = self.default_hint.clone();
            &hint_storage
        };

        match self.strategy {
            Strategy::Adaptive => self.apply_adaptive(args, hint_slice, max_iterations, policy),
            Strategy::Baseline => self.apply_baseline(args, hint_slice, policy),
        }
    }

    fn apply_adaptive(
        &mut self,
        args: &[Input],
        hints: &[Hint],
        max_iterations: usize,
        policy: OutputPolicy,
    ) -> Result<Vec<Ival>, RivalError> {
        for iteration in 0..max_iterations {
            if let Some(results) = self.run_iteration(args, iteration, hints, policy)? {
                return Ok(results);
            }
        }

        Err(RivalError::Unsamplable)
    }

    fn apply_baseline(
        &mut self,
        args: &[Input],
        hints: &[Hint],
        policy: OutputPolicy,
    ) -> Result<Vec<Ival>, RivalError> {
        let mut prec = self.disc.target().saturating_add(10);
        let mut iter: usize = 0;

        loop {
            self.iteration = iter;
            self.baseline_adjust(prec);
            self.run_with_hint(args, hints)?;

            match self.collect_outputs(policy)? {
                Some(outputs) => return Ok(outputs),
                None => {
                    let next = prec.saturating_mul(2);
                    if next > self.max_precision {
                        return Err(RivalError::Unsamplable);
                    }
                    prec = next;
                    iter = iter.saturating_add(1);
                }
            }
        }
    }

    /// Run a single iteration with precision tuning and hint-guided evaluation.
    pub(crate) fn run_iteration(
        &mut self,
        args: &[Input],
        iteration: usize,
        hints: &[Hint],
        policy: OutputPolicy,
    ) -> Result<Option<Vec<Ival>>, RivalError> {
        assert_eq!(hints.len(), self.instructions.len(), "hint length mismatch");
        self.iteration = iteration;
        if self.adjust(hints) {
            return Err(RivalError::Unsamplable);
        }
        self.run_with_hint(args, hints)?;
        self.collect_outputs(policy)
    }

    /// Analyze an input rectangle.
    ///
    /// Returns a `(status, hints, converged)` tuple:
    ///
    /// - `status` describes failure under the selected output policy.
    ///   A true lower endpoint means definite failure; otherwise, a true
    ///   upper endpoint means possible failure.
    ///
    /// - `hints` is a vector of [`Hint`]s that can be passed to
    ///   subsequent calls to [`Machine::apply`] to skip unnecessary
    ///   computation.
    ///
    /// - `converged` indicates whether the analysis has converged.
    pub fn analyze_with_hints(
        &mut self,
        rect: &[Ival],
        hint: Option<&[Hint]>,
        policy: OutputPolicy,
    ) -> (Ival, Vec<Hint>, bool) {
        let inputs: Vec<Input> = rect.iter().cloned().map(Input::Interval).collect();
        self.reset_evaluation(inputs.len());

        // Use provided hint or default.
        let tmp;
        let hint_slice = if let Some(h) = hint {
            h
        } else {
            tmp = self.default_hint.clone();
            &tmp
        };

        // One analysis iteration at sampling iteration 0.
        self.iteration = 0;
        match self.strategy {
            Strategy::Adaptive => {
                self.adjust(hint_slice);
            }
            Strategy::Baseline => self.baseline_adjust(self.disc.target().saturating_add(10)),
        }
        self.run_with_hint(&inputs, hint_slice)
            .expect("interval inputs do not require parsing");

        let (good, _done, bad, stuck) = self.return_flags(policy);
        let (next_hint, converged) = self.make_hint(hint_slice);

        let status = Ival::bool_interval(bad || stuck, (!good) || stuck);
        (status, next_hint, converged)
    }

    /// Analyze an input rectangle whose endpoints may be exact typed values.
    /// Endpoints are enclosed at the machine's argument precision before the
    /// existing single-pass analysis. Outward rounding may yield `MaybeInvalid`
    /// for a rectangle that a tighter enclosure could prove valid.
    pub fn analyze_inputs_with_hints(
        &mut self,
        rect: &[InputRange],
        hint: Option<&[Hint]>,
        policy: OutputPolicy,
    ) -> Result<(Ival, Vec<Hint>, bool), RivalError> {
        assert_eq!(rect.len(), self.arguments.len(), "Argument count mismatch");
        let precision = self.argument_precision();
        let rect = rect
            .iter()
            .enumerate()
            .map(|(index, range)| {
                range
                    .to_ival(precision)
                    .map_err(|source| RivalError::InputParse { index, source })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self.analyze_with_hints(&rect, hint, policy))
    }

    /// Analyze an input rectangle with typed lower and upper endpoints.
    pub fn analyze_inputs(
        &mut self,
        rect: &[InputRange],
        policy: OutputPolicy,
    ) -> Result<Ival, RivalError> {
        self.analyze_inputs_with_hints(rect, None, policy)
            .map(|(status, _, _)| status)
    }

    /// Analyze a hyper-rectangle and return only the boolean interval status.
    ///
    /// Returns the status described by [`Machine::analyze_with_hints`].
    ///
    /// The advantage of `analyze` over `apply` is that it applies to
    /// whole ranges of input points and is much faster.
    pub fn analyze(&mut self, rect: &[Ival], policy: OutputPolicy) -> Ival {
        let (status, _hint, _conv) = self.analyze_with_hints(rect, None, policy);
        status
    }

    /// Load interval arguments into their existing registers. Typed values are
    /// borrowed by the conversion instructions until evaluation finishes.
    fn reset_evaluation(&mut self, argument_count: usize) {
        assert_eq!(
            argument_count,
            self.arguments.len(),
            "Argument count mismatch"
        );
        self.bumps = 0;
        self.bumps_activated = false;
        self.iteration = 0;
        self.precisions.fill(0);
        self.repeats.fill(false);
        self.output_distance.fill(false);
        self.profiler.reset();
    }

    /// Execute instructions once using the supplied precision and hint plan.
    fn run_with_hint(&mut self, args: &[Input], hints: &[Hint]) -> Result<(), RivalError> {
        assert_eq!(hints.len(), self.instructions.len(), "hint length mismatch");
        // On the first iteration use the initial plan; subsequent iterations use tuned state.
        let (precisions, repeats) = if self.iteration == 0 {
            (&self.initial_precisions[..], &self.initial_repeats[..])
        } else {
            (&self.precisions[..], &self.repeats[..])
        };

        for (idx, (instruction, &repeat, &precision, hint)) in
            enumerate(izip!(&self.instructions, repeats, precisions, hints))
        {
            if repeat {
                continue;
            }
            let out_reg = self.instruction_register(idx);

            // Hints can override execution.
            match hint {
                Hint::Skip => {}
                Hint::Execute => {
                    if self.profiling_enabled {
                        let start = Instant::now();
                        execute::evaluate_instruction(
                            instruction,
                            &mut self.registers,
                            precision,
                            args,
                        )?;
                        let dt = start.elapsed().as_secs_f64() * 1000.0;
                        let exec = Execution {
                            name: instruction.data.name_static(),
                            number: idx as i32,
                            precision,
                            time_ms: dt,
                            iteration: self.iteration,
                        };
                        self.profiler.record(exec);
                    } else {
                        execute::evaluate_instruction(
                            instruction,
                            &mut self.registers,
                            precision,
                            args,
                        )?;
                    }
                }
                // Path reduction aliasing the output of an instruction to one of its inputs.
                Hint::Alias(pos) => {
                    if let Some(src_reg) = instruction.data.input_at(*pos as usize)
                        && src_reg != out_reg
                    {
                        let (src, dst) = if src_reg < out_reg {
                            let (left, right) = self.registers.split_at_mut(out_reg);
                            (&left[src_reg], &mut right[0])
                        } else {
                            let (left, right) = self.registers.split_at_mut(src_reg);
                            (&right[0], &mut left[out_reg])
                        };
                        dst.assign_from(src);
                    }
                }
                // Use pre-computed boolean value.
                Hint::KnownBool(value) => {
                    self.registers[out_reg] = Ival::bool_interval(*value, *value);
                }
            }
        }
        Ok(())
    }

    fn baseline_adjust(&mut self, new_prec: u32) {
        let instruction_count = self.instructions.len();
        let profiling = self.profiling_enabled;
        let start_time = if profiling {
            Some(Instant::now())
        } else {
            None
        };

        // Baseline uses a single global precision for all instructions.
        self.precisions.fill(new_prec);

        if self.iteration != 0 {
            // Determine which instructions can affect outputs (must be executed).
            let mut useful = vec![false; instruction_count];
            for &root in &self.outputs {
                if let Some(idx) = self.register_to_instruction(root) {
                    useful[idx] = true;
                }
            }

            for idx in (0..instruction_count).rev() {
                if !useful[idx] {
                    continue;
                }
                let out_reg = self.instruction_register(idx);
                let reg = &self.registers[out_reg];
                if reg.lo.immovable && reg.hi.immovable {
                    useful[idx] = false;
                    continue;
                }
                self.instructions[idx].for_each_input(|reg| {
                    useful[reg] = true;
                });
            }

            // Set repeats and update constant precisions.
            for (idx, &is_useful) in useful.iter().enumerate() {
                let is_constant = self.initial_repeats[idx];
                let best_known = self.best_known_precisions[idx];

                let mut inputs_stable = true;
                if is_constant {
                    self.instructions[idx].for_each_input(|reg| {
                        if !self.repeats[reg] {
                            inputs_stable = false;
                        }
                    });
                }

                let no_need_to_reevaluate = is_constant && new_prec <= best_known && inputs_stable;
                let result_is_exact_already = !is_useful;
                let repeat = result_is_exact_already || no_need_to_reevaluate;

                if is_constant && !repeat {
                    self.best_known_precisions[idx] = new_prec;
                }
                self.repeats[idx] = repeat;
            }
        }

        if profiling && let Some(t0) = start_time {
            let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
            self.profiler.record(Execution {
                name: "adjust",
                number: -1,
                precision: (self.iteration as u32) * 1000,
                time_ms: dt_ms,
                iteration: self.iteration,
            });
        }
    }

    /// Gather outputs and translate evaluation state into convergence results.
    fn collect_outputs(&mut self, policy: OutputPolicy) -> Result<Option<Vec<Ival>>, RivalError> {
        let (good, done, bad, stuck) = self.return_flags(policy);
        let mut outputs = Vec::with_capacity(self.outputs.len());

        for &root in &self.outputs {
            outputs.push(self.registers[root].clone());
        }

        if bad {
            return Err(RivalError::InvalidInput);
        }
        if done && good {
            return Ok(Some(outputs));
        }
        if stuck {
            return Err(RivalError::Unsamplable);
        }

        Ok(None)
    }

    /// Compute (good, done, bad, stuck) flags and update output_distance.
    fn return_flags(&mut self, policy: OutputPolicy) -> (bool, bool, bool, bool) {
        let require_all = policy == OutputPolicy::RequireAll;
        let mut good = require_all || self.outputs.is_empty();
        let mut done = true;
        let mut bad = !require_all && !self.outputs.is_empty();
        let mut stuck = false;

        for (idx, &root) in self.outputs.iter().enumerate() {
            let value = &self.registers[root];
            if require_all {
                if value.err.total {
                    bad = true;
                } else if value.err.partial {
                    good = false;
                }
            } else {
                good |= !value.err.partial && !value.err.total;
                bad &= value.err.total;
            }
            self.output_distance[idx] = false;

            if !require_all && value.err.total {
                continue;
            }
            if !require_all && value.err.partial {
                done = false;
            }

            let lo = self.disc.convert(idx, value.lo.as_float());
            let hi = self.disc.convert(idx, value.hi.as_float());
            let dist = self.disc.distance(idx, &lo, &hi);
            self.output_distance[idx] = dist == 1;
            if dist != 0 {
                done = false;
                if value.lo.immovable && value.hi.immovable {
                    stuck = true;
                }
            }
        }

        (good, done, bad, stuck)
    }
}

/// Errors that can occur during [`Machine::apply`].
///
/// Note that [`Machine::apply`] will only return a result if it can prove
/// that it has correctly rounded every non-error output. It only returns
/// [`RivalError::InvalidInput`] when the selected evaluation policy proves
/// the input invalid.
#[derive(Error, Debug)]
pub enum RivalError {
    /// A runtime string could not be parsed; `index` is its argument position.
    #[error("Could not parse input {index}: {source}")]
    InputParse {
        index: usize,
        source: crate::ParseIvalError,
    },
    /// The input point is invalid under the evaluation policy being used.
    ///
    /// For example, taking the square root of a negative number, or
    /// dividing by zero.
    #[error("Invalid input for rival machine")]
    InvalidInput,
    /// Rival was unable to correctly round the output within the
    /// configured precision and iteration limits.
    #[error("Unsamplable input for rival machine")]
    Unsamplable,
}
