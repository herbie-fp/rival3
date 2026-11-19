//! Precision-trace utilities for debugging adaptive tuning decisions.

use std::fmt;

use crate::{
    RivalError,
    eval::{
        adjust::path_reduction::path_reduction,
        instructions::InstructionData,
        machine::{Discretization, Hint, Machine},
        tricks::{AmplBounds, TrickContext, clamp_to_bits, slack_bits},
    },
    interval::Ival,
};

/// High-level mode used for a precision-tuning iteration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrecisionBreakdownMode {
    /// Iteration 0 uses the static seed plan computed at compile time.
    Seed,
    /// Adaptive tuning used the standard amplification pass.
    Adaptive,
    /// Adaptive tuning used bumps mode.
    Bumps,
}

impl fmt::Display for PrecisionBreakdownMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            PrecisionBreakdownMode::Seed => "seed",
            PrecisionBreakdownMode::Adaptive => "adaptive",
            PrecisionBreakdownMode::Bumps => "bumps",
        };
        f.write_str(label)
    }
}

/// Precision decision for one instruction in one iteration.
#[derive(Clone, Debug)]
pub struct PrecisionBreakdownRow {
    /// Instruction index in the compiled machine.
    pub instruction: usize,
    /// Static operation name.
    pub name: String,
    /// Precision in bits used for this node in this iteration.
    pub precision: u32,
    /// Whether the interval operator was actually executed.
    pub executed: bool,
    /// Human-readable explanation of how this precision was chosen.
    pub explanation: String,
}

/// Precision decisions for a single machine iteration.
#[derive(Clone, Debug)]
pub struct PrecisionBreakdownIteration {
    /// Sampling iteration number.
    pub iteration: usize,
    /// High-level tuning mode for this iteration.
    pub mode: PrecisionBreakdownMode,
    /// Extra note about the iteration-wide behavior.
    pub note: String,
    /// One row per machine instruction.
    pub rows: Vec<PrecisionBreakdownRow>,
}

#[derive(Clone)]
struct Candidate {
    value: u32,
    description: String,
}

struct PassTrace {
    precisions: Vec<u32>,
    candidates: Vec<Vec<Candidate>>,
}

struct UpdateTrace {
    repeats: Vec<bool>,
    displayed_bits: Vec<u32>,
    executed: Vec<bool>,
    overrides: Vec<Option<String>>,
    suffixes: Vec<String>,
    any_reevaluation: bool,
}

struct AdjustTrace {
    iteration: PrecisionBreakdownIteration,
    unsamplable: bool,
}

impl<D: Discretization> Machine<D> {
    /// Re-evaluate the machine and return a per-iteration precision trace.
    ///
    /// This follows the same adaptive tuning loop as [`Machine::apply`], so it
    /// resets the machine state and profiling buffer before running.
    pub fn explain_precision_iterations(
        &mut self,
        args: &[Ival],
        hint: Option<&[Hint]>,
    ) -> Result<Vec<PrecisionBreakdownIteration>, RivalError> {
        self.load_arguments(args);

        let hint_storage;
        let hints = if let Some(h) = hint {
            h
        } else {
            hint_storage = self.default_hint.clone();
            &hint_storage
        };
        assert_eq!(hints.len(), self.instructions.len(), "hint length mismatch");

        let mut trace = vec![trace_seed_iteration(self, hints)];
        let mut iteration = 0usize;

        loop {
            self.iteration = iteration;
            if iteration > 0 {
                let adjust_trace = trace_backward_pass(self, hints);
                trace.push(adjust_trace.iteration);
                if adjust_trace.unsamplable {
                    return Err(RivalError::Unsamplable);
                }
            }

            self.run_with_hint(hints);
            if self.collect_outputs()?.is_some() {
                return Ok(trace);
            }
            iteration = iteration.saturating_add(1);
        }
    }
}

fn trace_seed_iteration<D: Discretization>(
    machine: &Machine<D>,
    hints: &[Hint],
) -> PrecisionBreakdownIteration {
    let instruction_count = machine.instructions.len();
    let mut seed_precisions = vec![0u32; instruction_count];
    let mut candidates = vec![Vec::<Candidate>::new(); instruction_count];

    let target_seed = machine
        .disc
        .target()
        .saturating_add(machine.base_tuning_precision);
    for &root in &machine.outputs {
        if let Some(idx) = machine.register_to_instruction(root) {
            seed_precisions[idx] = target_seed;
            candidates[idx].push(Candidate {
                value: target_seed,
                description: format!(
                    "target({}) + base_tuning_precision({}) = {}",
                    machine.disc.target(),
                    machine.base_tuning_precision,
                    target_seed
                ),
            });
        }
    }

    for idx in (0..instruction_count).rev() {
        let current = seed_precisions[idx];
        if current == 0 {
            continue;
        }

        machine.instructions[idx].for_each_input(|reg| {
            if let Some(child_idx) = machine.register_to_instruction(reg) {
                if child_idx == idx {
                    return;
                }
                let propagated = current.saturating_add(machine.ampl_tuning_bits);
                candidates[child_idx].push(Candidate {
                    value: propagated,
                    description: format!(
                        "from {} + ampl_tuning_bits({}) = {}",
                        node_tag(machine, idx),
                        machine.ampl_tuning_bits,
                        propagated
                    ),
                });
                seed_precisions[child_idx] = seed_precisions[child_idx].max(propagated);
            }
        });
    }

    debug_assert_eq!(seed_precisions, machine.initial_precisions);

    let mut rows = Vec::with_capacity(instruction_count);
    for idx in 0..instruction_count {
        let precision = machine.initial_precisions[idx];
        let executed = !machine.initial_repeats[idx] && matches!(hints[idx], Hint::Execute);
        let mut explanation = format_seed_reason(
            machine,
            idx,
            precision,
            &candidates[idx],
            machine.initial_repeats[idx],
        );
        explanation.push_str(&hint_suffix(&hints[idx]));
        rows.push(PrecisionBreakdownRow {
            instruction: idx,
            name: machine.instructions[idx].data.name_static().to_string(),
            precision,
            executed,
            explanation,
        });
    }

    PrecisionBreakdownIteration {
        iteration: 0,
        mode: PrecisionBreakdownMode::Seed,
        note: format!(
            "seed plan from target({}), base_tuning_precision({}), ampl_tuning_bits({})",
            machine.disc.target(),
            machine.base_tuning_precision,
            machine.ampl_tuning_bits
        ),
        rows,
    }
}

fn trace_backward_pass<D: Discretization>(machine: &mut Machine<D>, hints: &[Hint]) -> AdjustTrace {
    let instruction_count = machine.instructions.len();
    let first_tuning_pass = machine.iteration == 1;
    let previously_using_bumps = machine.bumps_activated;

    let mut root_extras = vec![0u32; instruction_count];
    let mut work_repeats = vec![true; instruction_count];

    let slack = slack_bits(machine.iteration, machine.slack_unit);
    let mut boundary_hits = 0usize;
    for (&root, &boundary_issue) in machine.outputs.iter().zip(machine.output_distance.iter()) {
        if boundary_issue && let Some(idx) = machine.register_to_instruction(root) {
            root_extras[idx] = root_extras[idx].max(slack);
            boundary_hits = boundary_hits.saturating_add(1);
        }
    }

    for &root in &machine.outputs {
        if let Some(idx) = machine.register_to_instruction(root) {
            work_repeats[idx] = false;
        }
    }

    for idx in (0..instruction_count).rev() {
        if work_repeats[idx] {
            continue;
        }

        let reg = &machine.registers[machine.instruction_register(idx)];
        if reg.lo.immovable && reg.hi.immovable {
            work_repeats[idx] = true;
            continue;
        }

        let var_count = machine.arguments.len();
        let mut mark = |reg: usize| {
            if reg >= var_count {
                work_repeats[reg - var_count] = false;
            }
        };
        path_reduction(machine, idx, &mut mark);
    }

    let normal_pass = trace_tuning_pass(
        machine,
        hints,
        &work_repeats,
        root_extras,
        previously_using_bumps,
    );
    let normal_update = trace_repeat_updates(
        machine,
        hints,
        &work_repeats,
        &normal_pass.precisions,
        first_tuning_pass,
    );

    let use_bumps_this_iteration = previously_using_bumps || !normal_update.any_reevaluation;
    let fallback_triggered = !previously_using_bumps && !normal_update.any_reevaluation;

    let (mode, note, pass, update, unsamplable) = if normal_update.any_reevaluation {
        (
            if previously_using_bumps {
                PrecisionBreakdownMode::Bumps
            } else {
                PrecisionBreakdownMode::Adaptive
            },
            make_iteration_note(previously_using_bumps, false, boundary_hits, slack),
            normal_pass,
            normal_update,
            false,
        )
    } else {
        machine.bumps = machine.bumps.saturating_add(1);
        machine.bumps_activated = true;

        let bump_repeats = vec![false; instruction_count];
        let bump_pass = trace_tuning_pass(
            machine,
            hints,
            &bump_repeats,
            vec![0u32; instruction_count],
            true,
        );
        let bump_update = trace_repeat_updates(
            machine,
            hints,
            &bump_repeats,
            &bump_pass.precisions,
            first_tuning_pass,
        );
        let stuck = !bump_update.any_reevaluation;
        (
            PrecisionBreakdownMode::Bumps,
            make_iteration_note(
                use_bumps_this_iteration,
                fallback_triggered,
                boundary_hits,
                slack,
            ),
            bump_pass,
            bump_update,
            stuck,
        )
    };

    if !unsamplable {
        machine.repeats.copy_from_slice(&update.repeats);
        machine.precisions.copy_from_slice(&pass.precisions);
        if matches!(mode, PrecisionBreakdownMode::Bumps) {
            machine.bumps_activated = true;
        }
    }

    let rows = (0..instruction_count)
        .map(|idx| {
            let mut explanation = format_dynamic_reason(
                machine,
                idx,
                &pass.candidates[idx],
                &update.overrides[idx],
                &update.suffixes[idx],
            );
            explanation.push_str(&hint_suffix(&hints[idx]));
            PrecisionBreakdownRow {
                instruction: idx,
                name: machine.instructions[idx].data.name_static().to_string(),
                precision: update.displayed_bits[idx],
                executed: update.executed[idx],
                explanation,
            }
        })
        .collect();

    AdjustTrace {
        iteration: PrecisionBreakdownIteration {
            iteration: machine.iteration,
            mode,
            note,
            rows,
        },
        unsamplable,
    }
}

fn trace_tuning_pass<D: Discretization>(
    machine: &Machine<D>,
    hints: &[Hint],
    repeats: &[bool],
    mut initial_extras: Vec<u32>,
    bumps_enabled: bool,
) -> PassTrace {
    let instruction_count = machine.instructions.len();
    let mut candidates = vec![Vec::<Candidate>::new(); instruction_count];
    let mut vprecs_min = vec![0u32; instruction_count];

    for idx in 0..instruction_count {
        if initial_extras[idx] > 0 {
            candidates[idx].push(Candidate {
                value: initial_extras[idx],
                description: format!("root boundary slack = {}", initial_extras[idx]),
            });
        }
    }

    let ctx = TrickContext::new(
        machine.iteration,
        machine.lower_bound_early_stopping,
        bumps_enabled,
        machine.slack_unit,
    );

    for idx in (0..instruction_count).rev() {
        if repeats[idx] || matches!(hints[idx], Hint::Skip) {
            continue;
        }

        let instruction = &machine.instructions[idx];
        let output = &machine.registers[machine.instruction_register(idx)];
        let parent_upper = initial_extras[idx];
        let parent_lower = vprecs_min[idx];
        let base_precision = machine.initial_precisions[idx];

        initial_extras[idx] = base_precision
            .saturating_add(parent_upper)
            .clamp(machine.min_precision, machine.max_precision);

        match &instruction.data {
            InstructionData::Literal { .. }
            | InstructionData::Rational { .. }
            | InstructionData::Constant { .. } => {}
            InstructionData::Unary { op, arg } => {
                let bounds = ctx.bounds_for_unary(*op, output, &machine.registers[*arg]);
                propagate_candidate(
                    machine,
                    *arg,
                    parent_upper,
                    parent_lower,
                    bounds,
                    &mut initial_extras,
                    &mut vprecs_min,
                    &mut candidates,
                    idx,
                    "arg",
                );
            }
            InstructionData::UnaryParam { op, param, arg } => {
                let bounds =
                    ctx.bounds_for_unary_param(*op, *param, output, &machine.registers[*arg]);
                propagate_candidate(
                    machine,
                    *arg,
                    parent_upper,
                    parent_lower,
                    bounds,
                    &mut initial_extras,
                    &mut vprecs_min,
                    &mut candidates,
                    idx,
                    "arg",
                );
            }
            InstructionData::Binary { op, lhs, rhs } => {
                let (lhs_bounds, rhs_bounds) = ctx.bounds_for_binary(
                    *op,
                    output,
                    &machine.registers[*lhs],
                    &machine.registers[*rhs],
                );
                propagate_candidate(
                    machine,
                    *lhs,
                    parent_upper,
                    parent_lower,
                    lhs_bounds,
                    &mut initial_extras,
                    &mut vprecs_min,
                    &mut candidates,
                    idx,
                    "lhs",
                );
                propagate_candidate(
                    machine,
                    *rhs,
                    parent_upper,
                    parent_lower,
                    rhs_bounds,
                    &mut initial_extras,
                    &mut vprecs_min,
                    &mut candidates,
                    idx,
                    "rhs",
                );
            }
            InstructionData::Ternary {
                op,
                arg1,
                arg2,
                arg3,
            } => {
                let (bounds1, bounds2, bounds3) = ctx.bounds_for_ternary(
                    *op,
                    output,
                    &machine.registers[*arg1],
                    &machine.registers[*arg2],
                    &machine.registers[*arg3],
                );
                propagate_candidate(
                    machine,
                    *arg1,
                    parent_upper,
                    parent_lower,
                    bounds1,
                    &mut initial_extras,
                    &mut vprecs_min,
                    &mut candidates,
                    idx,
                    "arg1",
                );
                propagate_candidate(
                    machine,
                    *arg2,
                    parent_upper,
                    parent_lower,
                    bounds2,
                    &mut initial_extras,
                    &mut vprecs_min,
                    &mut candidates,
                    idx,
                    "arg2",
                );
                propagate_candidate(
                    machine,
                    *arg3,
                    parent_upper,
                    parent_lower,
                    bounds3,
                    &mut initial_extras,
                    &mut vprecs_min,
                    &mut candidates,
                    idx,
                    "arg3",
                );
            }
        }
    }

    PassTrace {
        precisions: initial_extras,
        candidates,
    }
}

fn propagate_candidate<D: Discretization>(
    machine: &Machine<D>,
    child_reg: usize,
    parent_upper: u32,
    parent_lower: u32,
    bounds: AmplBounds,
    vprecs_max: &mut [u32],
    vprecs_min: &mut [u32],
    candidates: &mut [Vec<Candidate>],
    parent_idx: usize,
    edge: &'static str,
) {
    if let Some(child_idx) = machine.register_to_instruction(child_reg) {
        let candidate_upper = clamp_to_bits((parent_upper as i64).saturating_add(bounds.upper));
        let candidate_lower =
            clamp_to_bits((parent_lower as i64).saturating_add(bounds.lower.max(0)));
        vprecs_max[child_idx] = vprecs_max[child_idx].max(candidate_upper);
        vprecs_min[child_idx] = vprecs_min[child_idx].max(candidate_lower);
        candidates[child_idx].push(Candidate {
            value: candidate_upper,
            description: format!(
                "from {}/{} = {} (parent_extra {} + ampl_upper {})",
                node_tag(machine, parent_idx),
                edge,
                candidate_upper,
                parent_upper,
                bounds.upper
            ),
        });
    }
}

fn trace_repeat_updates<D: Discretization>(
    machine: &mut Machine<D>,
    hints: &[Hint],
    pre_tuning_repeats: &[bool],
    new_precisions: &[u32],
    first_tuning_pass: bool,
) -> UpdateTrace {
    let instruction_count = machine.instructions.len();
    let mut repeats = pre_tuning_repeats.to_vec();
    let mut displayed_bits = vec![0u32; instruction_count];
    let mut executed = vec![false; instruction_count];
    let mut overrides = vec![None; instruction_count];
    let mut suffixes = vec![String::new(); instruction_count];
    let old_precisions: &[u32] = if first_tuning_pass {
        &machine.initial_precisions
    } else {
        &machine.precisions
    };
    let var_count = machine.arguments.len();
    let mut any_reevaluation = false;

    for idx in 0..instruction_count {
        let old_precision = old_precisions[idx];
        let stored_precision = if machine.initial_repeats[idx] {
            machine.best_known_precisions[idx].max(old_precision)
        } else {
            old_precision
        };

        if repeats[idx] {
            displayed_bits[idx] = stored_precision;
            executed[idx] = false;
            overrides[idx] = Some(if machine.initial_repeats[idx] {
                format!(
                    "stored constant precision {stored_precision}; exact value reused without reevaluation"
                )
            } else {
                let reg = &machine.registers[machine.instruction_register(idx)];
                if reg.lo.immovable && reg.hi.immovable {
                    format!(
                        "stored precision {stored_precision}; previous result already exact, so no reevaluation was requested"
                    )
                } else {
                    format!(
                        "stored precision {stored_precision}; subgraph was not scheduled for reevaluation"
                    )
                }
            });
            continue;
        }

        let reference = if machine.initial_repeats[idx] {
            machine.best_known_precisions[idx]
        } else {
            old_precision
        };

        let self_reg = machine.instruction_register(idx);
        let mut children_repeat = true;
        machine.instructions[idx].data.for_each_input(|reg| {
            if reg != self_reg && reg >= var_count && !repeats[reg - var_count] {
                children_repeat = false;
            }
        });

        let new_precision = new_precisions[idx];
        let precision_has_not_increased = new_precision <= reference && children_repeat;
        let precision_has_increased = !precision_has_not_increased;
        any_reevaluation |= precision_has_increased;

        if machine.initial_repeats[idx] && precision_has_increased {
            machine.best_known_precisions[idx] = new_precision;
        }

        repeats[idx] = precision_has_not_increased;
        displayed_bits[idx] = if repeats[idx] {
            reference
        } else {
            new_precision
        };
        executed[idx] = !repeats[idx] && matches!(hints[idx], Hint::Execute);
        suffixes[idx] = if repeats[idx] {
            format!(
                "; reused previous result (new {} <= reference {}, children stable)",
                new_precision, reference
            )
        } else if new_precision > reference {
            format!(
                "; reevaluated because {} > reference {}",
                new_precision, reference
            )
        } else {
            format!(
                "; reevaluated because a child changed at the same precision {}",
                new_precision
            )
        };
    }

    UpdateTrace {
        repeats,
        displayed_bits,
        executed,
        overrides,
        suffixes,
        any_reevaluation,
    }
}

fn format_seed_reason<D: Discretization>(
    machine: &Machine<D>,
    idx: usize,
    precision: u32,
    candidates: &[Candidate],
    cached_constant: bool,
) -> String {
    let mut reason = if candidates.is_empty() {
        format!("seed({precision})")
    } else if candidates.len() == 1 {
        candidates[0].description.clone()
    } else {
        format!("max({}) = {precision}", format_candidates(candidates))
    };

    if cached_constant {
        reason.push_str("; constant node cached after compilation");
    } else if idx == machine.outputs[0].saturating_sub(machine.arguments.len()) {
        reason.push_str("; root seed precision");
    }

    reason
}

fn format_dynamic_reason<D: Discretization>(
    machine: &Machine<D>,
    idx: usize,
    candidates: &[Candidate],
    override_reason: &Option<String>,
    suffix: &str,
) -> String {
    if let Some(reason) = override_reason {
        return reason.clone();
    }

    let seed = machine.initial_precisions[idx];
    let extra = candidates
        .iter()
        .map(|candidate| candidate.value)
        .max()
        .unwrap_or(0);
    let final_bits = seed
        .saturating_add(extra)
        .clamp(machine.min_precision, machine.max_precision);

    let mut reason = if candidates.is_empty() {
        format!("seed({seed}) + extra(0) = {final_bits}")
    } else {
        format!(
            "seed({seed}) + max({}) = {final_bits}",
            format_candidates(candidates)
        )
    };
    reason.push_str(suffix);
    reason
}

fn format_candidates(candidates: &[Candidate]) -> String {
    candidates
        .iter()
        .map(|candidate| candidate.description.clone())
        .collect::<Vec<_>>()
        .join(", ")
}

fn make_iteration_note(
    bumps_active: bool,
    fallback_triggered: bool,
    boundary_hits: usize,
    slack: u32,
) -> String {
    let mut notes = Vec::new();
    if boundary_hits > 0 {
        notes.push(format!(
            "root boundary slack({slack}) applied to {boundary_hits} output(s)"
        ));
    }
    if fallback_triggered {
        notes.push(
            "normal pass made no progress, so Rival reran this iteration in bumps mode".to_string(),
        );
    } else if bumps_active {
        notes.push("bumps mode remained active from an earlier iteration".to_string());
    }
    notes.join("; ")
}

fn node_tag<D: Discretization>(machine: &Machine<D>, idx: usize) -> String {
    format!("{}#{}", machine.instructions[idx].data.name_static(), idx)
}

fn hint_suffix(hint: &Hint) -> String {
    match hint {
        Hint::Execute => String::new(),
        Hint::Skip => "; skipped by hint".to_string(),
        Hint::Alias(pos) => format!("; aliased to input {}", pos),
        Hint::KnownBool(value) => format!("; resolved to known bool {}", value),
    }
}
