//! Main evaluation loop with adaptive precision tuning

use itertools::{enumerate, izip};

use crate::eval::{
    execute,
    machine::{Discretization, Hint, Machine},
    profile::Execution,
};
use crate::interval::Ival;

impl<D: Discretization> Machine<D> {
    /// Evaluate the machine with given inputs until convergence or the iteration limit
    pub fn apply(
        &mut self,
        args: &[Ival],
        hint: Option<&[Hint]>,
        max_iterations: usize,
    ) -> Result<Vec<Ival>, RivalError> {
        self.load_arguments(args);
        let hint_storage;
        let hint_slice: &[Hint] = if let Some(h) = hint {
            h
        } else {
            hint_storage = self.default_hint.clone();
            &hint_storage
        };

        // Run iterations [0, max_iterations]
        for iteration in 0..=max_iterations {
            if let Some(results) = self.run_iteration(iteration, hint_slice)? {
                return Ok(results);
            }
        }

        Err(RivalError::Unsamplable)
    }

    /// Evaluate the machine using the baseline strategy
    pub fn apply_baseline(
        &mut self,
        args: &[Ival],
        hint: Option<&[Hint]>,
    ) -> Result<Vec<Ival>, RivalError> {
        self.load_arguments(args);

        let hint_storage;
        let hint_slice: &[Hint] = if let Some(h) = hint {
            h
        } else {
            hint_storage = self.default_hint.clone();
            &hint_storage
        };

        let start_prec = self.disc.target().saturating_add(10);
        let mut prec = start_prec;
        let mut iter: usize = 0;

        loop {
            self.iteration = iter;
            self.baseline_adjust(prec);
            self.run_with_hint(hint_slice);

            match self.collect_outputs()? {
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

    /// Analyze an input rectangle using the baseline strategy
    pub fn analyze_baseline_with_hints(
        &mut self,
        rect: &[Ival],
        hint: Option<&[Hint]>,
    ) -> (Ival, Vec<Hint>, bool) {
        self.load_arguments(rect);

        let tmp;
        let hint_slice = if let Some(h) = hint {
            h
        } else {
            tmp = self.default_hint.clone();
            &tmp
        };

        self.iteration = 0;
        self.baseline_adjust(self.disc.target().saturating_add(10));
        self.run_with_hint(hint_slice);

        let (good, _done, bad, stuck) = self.return_flags();
        let (next_hint, converged) = self.make_hint(hint_slice);

        let status = Ival::bool_interval(bad || stuck, !good);
        (status, next_hint, converged)
    }

    /// Analyze a hyper-rectangle using the baseline strategy and return only the boolean interval status.
    pub fn analyze_baseline(&mut self, rect: &[Ival]) -> Ival {
        let (status, _hint, _conv) = self.analyze_baseline_with_hints(rect, None);
        status
    }

    /// Run a single iteration with precision tuning and hint-guided evaluation
    pub fn run_iteration(
        &mut self,
        iteration: usize,
        hints: &[Hint],
    ) -> Result<Option<Vec<Ival>>, RivalError> {
        assert_eq!(hints.len(), self.instructions.len(), "hint length mismatch");
        self.iteration = iteration;
        if self.adjust(hints) {
            return Err(RivalError::Unsamplable);
        }
        self.run_with_hint(hints);
        self.collect_outputs()
    }

    /// Analyze an input rectangle and return status summary, next hints, and convergence flag
    pub fn analyze_with_hints(
        &mut self,
        rect: &[Ival],
        hint: Option<&[Hint]>,
    ) -> (Ival, Vec<Hint>, bool) {
        self.load_arguments(rect);

        // Use provided hint or default
        let tmp;
        let hint_slice = if let Some(h) = hint {
            h
        } else {
            tmp = self.default_hint.clone();
            &tmp
        };

        // One analysis iteration at sampling iteration 0
        self.iteration = 0;
        self.adjust(hint_slice);
        self.run_with_hint(hint_slice);

        let (good, _done, bad, stuck) = self.return_flags();
        let (next_hint, converged) = self.make_hint(hint_slice);

        let status = Ival::bool_interval(bad || stuck, !good);
        (status, next_hint, converged)
    }

    /// Analyze a hyper-rectangle and return only the boolean interval status
    pub fn analyze(&mut self, rect: &[Ival]) -> Ival {
        let (status, _hint, _conv) = self.analyze_with_hints(rect, None);
        status
    }

    /// Load argument intervals into the front of the register file
    pub fn load_arguments(&mut self, args: &[Ival]) {
        assert_eq!(args.len(), self.arguments.len(), "Argument count mismatch");
        for (i, arg) in args.iter().cloned().enumerate() {
            self.registers[i] = arg;
        }
        self.bumps = 0;
        self.bumps_activated = false;
        self.iteration = 0;
        self.precisions.fill(0);
        self.repeats.fill(false);
        self.output_distance.fill(false);
        if self.profiling_enabled {
            self.profiler.reset();
        }
    }

    /// Execute instructions once using the supplied precision and hint plan
    fn run_with_hint(&mut self, hints: &[Hint]) {
        // On the first iteration use the initial plan; subsequent iterations use tuned state
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

            // Hints can override execution
            match hint {
                Hint::Skip => {}
                Hint::Execute => {
                    if self.profiling_enabled {
                        let start = std::time::Instant::now();
                        execute::evaluate_instruction(instruction, &mut self.registers, precision);
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
                        execute::evaluate_instruction(instruction, &mut self.registers, precision)
                    }
                }
                // Path reduction aliasing the output of an instruction to one of its inputs
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
                // Use pre-computed boolean value
                Hint::KnownBool(value) => {
                    self.registers[out_reg] = Ival::bool_interval(*value, *value);
                }
            }
        }
    }

    fn baseline_adjust(&mut self, new_prec: u32) {
        let instruction_count = self.instructions.len();
        let profiling = self.profiling_enabled;
        let start_time = if profiling {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Baseline uses a single global precision for all instructions
        self.precisions.fill(new_prec);

        if self.iteration != 0 {
            let var_count = self.arguments.len();

            // Determine which instructions can affect outputs (must be executed)
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
                    if reg >= var_count {
                        useful[reg - var_count] = true;
                    }
                });
            }

            // Set repeats and update constant precisions
            for idx in 0..instruction_count {
                let is_constant = self.initial_repeats[idx];
                let best_known = self.best_known_precisions[idx];

                let mut inputs_stable = true;
                if is_constant {
                    self.instructions[idx].for_each_input(|reg| {
                        if reg >= var_count && !self.repeats[reg - var_count] {
                            inputs_stable = false;
                        }
                    });
                }

                let no_need_to_reevaluate =
                    is_constant && new_prec <= best_known && inputs_stable;
                let result_is_exact_already = !useful[idx];
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

    /// Gather outputs and translate evaluation state into convergence results
    fn collect_outputs(&mut self) -> Result<Option<Vec<Ival>>, RivalError> {
        let (good, done, bad, stuck) = self.return_flags();
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

    /// Compute (good, done, bad, stuck) flags and update output_distance like Racket's rival-machine-return
    fn return_flags(&mut self) -> (bool, bool, bool, bool) {
        let mut good = true;
        let mut done = true;
        let mut bad = false;
        let mut stuck = false;

        for (idx, &root) in self.outputs.iter().enumerate() {
            let value = &self.registers[root];
            if value.err.total {
                bad = true;
            } else if value.err.partial {
                good = false;
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

#[derive(thiserror::Error, Debug)]
pub enum RivalError {
    #[error("Invalid input for rival machine")]
    InvalidInput,
    #[error("Unsamplable input for rival machine")]
    Unsamplable,
}
