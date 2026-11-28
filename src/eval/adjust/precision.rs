//! Precision tuning logic using amplification bounds

use crate::eval::{
    instructions::InstructionData,
    machine::{Discretization, Hint, Machine},
    tricks::{TrickContext, clamp_to_bits},
};
use itertools::{enumerate, izip};

/// Compute precision targets for instructions based on amplification bounds and hints
/// Returns true when the optional upper-bound precision-cap early exit fires.
pub(super) fn precision_tuning<D: Discretization>(
    machine: &Machine<D>,
    hints: &[Hint],
    repeats: &[bool],
    vprecs: &mut [u32],
) -> bool {
    let ctx = TrickContext::new(
        machine.iteration,
        machine.bumps_activated,
        machine.slack_unit,
    );

    for idx in (0..machine.instructions.len()).rev() {
        if repeats[idx] || matches!(hints[idx], Hint::Skip) {
            continue;
        }

        let instruction = &machine.instructions[idx];
        let output = &machine.registers[idx];
        let parent_prec = vprecs[idx];
        let base_precision = machine.initial_precisions[idx];

        vprecs[idx] = base_precision
            .saturating_add(parent_prec)
            .saturating_add(clamp_to_bits(ctx.logspan(output)))
            .clamp(machine.min_precision, machine.max_precision);

        if machine.early_exit && vprecs[idx] == machine.max_precision {
            return true;
        }

        // Propagate precision requirements to children based on error amplification.
        match &instruction.data {
            InstructionData::WithVal { .. }
            | InstructionData::Literal { .. }
            | InstructionData::Rational { .. }
            | InstructionData::Constant { .. } => {}
            InstructionData::Unary { op, arg } => {
                let bounds = ctx.bounds_for_unary(*op, output, &machine.registers[*arg]);
                propagate_child(*arg, bounds, parent_prec, vprecs);
            }
            InstructionData::UnaryParam { op, param, arg } => {
                let bounds =
                    ctx.bounds_for_unary_param(*op, *param, output, &machine.registers[*arg]);
                propagate_child(*arg, bounds, parent_prec, vprecs);
            }
            InstructionData::Binary { op, lhs, rhs } => {
                let (lhs_bounds, rhs_bounds) = ctx.bounds_for_binary(
                    *op,
                    output,
                    &machine.registers[*lhs],
                    &machine.registers[*rhs],
                );
                propagate_child(*lhs, lhs_bounds, parent_prec, vprecs);
                propagate_child(*rhs, rhs_bounds, parent_prec, vprecs);
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
                propagate_child(*arg1, bounds1, parent_prec, vprecs);
                propagate_child(*arg2, bounds2, parent_prec, vprecs);
                propagate_child(*arg3, bounds3, parent_prec, vprecs);
            }
        }
    }
    false
}

/// Update a child register with propagated precision requirements
fn propagate_child(
    child_reg: usize,
    bounds: i64,
    parent_prec: u32,
    vprecs: &mut [u32],
) {
    // TODO: We actually don't need clamp_to_bits here-- we can simply cast as u32 because we
    // assume that all stored precisions are positive and a cast won't overflow; see if there's
    // any noticeable performance difference if we don't use clamp_to_bits (likely not significant).
    vprecs[child_reg] = clamp_to_bits(
        (vprecs[child_reg] as i64).max((parent_prec as i64).saturating_add(bounds)),
    );
}

/// Mark instructions that can skip reevaluation and report whether any work remains
pub(super) fn update_repeats<D: Discretization>(
    machine: &mut Machine<D>,
    repeats: &mut [bool],
    vprecs: &[u32],
    first_tuning_pass: bool,
) -> bool {
    let mut any_reevaluation = false;

    let old_precisions: &[u32] = if first_tuning_pass {
        &machine.initial_precisions
    } else {
        &machine.precisions
    };

    for (idx, (instr, &new_precision, &constant)) in enumerate(izip!(
        &machine.instructions,
        vprecs,
        &machine.initial_repeats
    )) {
        if repeats[idx] {
            continue;
        }

        let old_precision = old_precisions[idx];
        let reference = if constant {
            machine.best_known_precisions[idx]
        } else {
            old_precision
        };

        // Recompute if precision increases or if any child recomputes.
        let mut children_repeat = true;
        instr.data.for_each_input(|reg| {
            if !repeats[reg] {
                children_repeat = false;
            }
        });

        let precision_has_not_increased = new_precision <= reference && children_repeat;
        let precision_has_increased = !precision_has_not_increased;
        any_reevaluation |= precision_has_increased;

        if constant && precision_has_increased {
            machine.best_known_precisions[idx] = new_precision;
        }

        repeats[idx] = precision_has_not_increased;
    }

    any_reevaluation
}
