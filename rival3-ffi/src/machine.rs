use crate::discretization::{RivalDiscType, RivalDiscretization};
use crate::expr::{RivalExpr, RivalExprBuilder};
use crate::options::RivalMachineOptions;
use crate::profile::InstructionNames;
use crate::{
    DiscTypeCode, OutputPolicyCode, RivalError, c_slice, c_slice_mut, mpfr_ptr, mpfr_srcptr, status,
};
use gmp_mpfr_sys::mpfr::{self, mpfr_t};
use num_enum::TryFromPrimitive;
use rival::{Discretization, ErrorFlags, Hint, Ival, Machine, MachineBuilder, OutputPolicy};
use rug::Assign;

pub struct RivalMachine {
    pub(crate) machine: Machine<RivalDiscretization>,
    pub(crate) arg_buf: Vec<Ival>,
    pub(crate) instruction_names: InstructionNames,
    pub(crate) n_vars: usize,
    pub(crate) n_exprs: usize,
}

pub struct RivalHints {
    hints: Vec<Hint>,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, TryFromPrimitive)]
pub enum RivalOutputPolicy {
    RequireAll = 0,
    AllowPartial = 1,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RivalAnalysis {
    Valid = 0,
    MaybeInvalid = 1,
    Invalid = 2,
}

fn output_policy(code: OutputPolicyCode) -> Result<OutputPolicy, RivalError> {
    Ok(match RivalOutputPolicy::try_from(code)? {
        RivalOutputPolicy::RequireAll => OutputPolicy::RequireAll,
        RivalOutputPolicy::AllowPartial => OutputPolicy::AllowPartial,
    })
}

unsafe fn machine_hints<'a>(
    machine: &RivalMachine,
    hints: *const RivalHints,
) -> Result<Option<&'a [Hint]>, RivalError> {
    match unsafe { hints.as_ref() } {
        Some(hints) if hints.hints.len() != machine.machine.instruction_count() => {
            Err(RivalError::Misuse)
        }
        hints => Ok(hints.map(|hints| hints.hints.as_slice())),
    }
}

unsafe fn marshal_point_args(args: &[*const mpfr_t], buf: &mut [Ival]) -> Result<(), RivalError> {
    for (ival, &ptr) in buf.iter_mut().zip(args) {
        if ptr.is_null() {
            return Err(RivalError::Misuse);
        }
        let src_prec = unsafe { mpfr::get_prec(ptr) } as u32;
        ival.set_prec(src_prec);
        unsafe {
            mpfr::set(ival.lo_mut().as_raw_mut(), ptr, mpfr::rnd_t::RNDN);
            mpfr::set(ival.hi_mut().as_raw_mut(), ptr, mpfr::rnd_t::RNDN);
        }
        ival.set_immovable(true, true);
        let err = if ival.lo().is_finite() {
            ErrorFlags::none()
        } else {
            ErrorFlags::error()
        };
        ival.set_error_flags(err);
    }
    Ok(())
}

fn marshal_f64_args(args: &[f64], buf: &mut [Ival]) {
    for (ival, &x) in buf.iter_mut().zip(args) {
        ival.set_prec(f64::MANTISSA_DIGITS);
        ival.lo_mut().assign(x);
        ival.hi_mut().assign(x);
        ival.set_immovable(true, true);
        let err = if x.is_finite() {
            ErrorFlags::none()
        } else {
            ErrorFlags::error()
        };
        ival.set_error_flags(err);
    }
}

unsafe fn marshal_rect_args(rect: &[*const mpfr_t], buf: &mut [Ival]) -> Result<(), RivalError> {
    for (ival, bounds) in buf.iter_mut().zip(rect.chunks_exact(2)) {
        let (lo_ptr, hi_ptr) = (bounds[0], bounds[1]);
        if lo_ptr.is_null() || hi_ptr.is_null() {
            return Err(RivalError::Misuse);
        }

        let lo_prec = unsafe { mpfr::get_prec(lo_ptr) } as u32;
        let hi_prec = unsafe { mpfr::get_prec(hi_ptr) } as u32;
        ival.set_prec(lo_prec.max(hi_prec));

        unsafe {
            mpfr::set(ival.lo_mut().as_raw_mut(), lo_ptr, mpfr::rnd_t::RNDN);
            mpfr::set(ival.hi_mut().as_raw_mut(), hi_ptr, mpfr::rnd_t::RNDN);
        }

        let (lo, hi) = (ival.lo(), ival.hi());
        let fixed = lo == hi;
        let err = lo.is_nan() || hi.is_nan() || (fixed && lo.is_infinite());
        ival.set_immovable(fixed, fixed);
        let flags = if err {
            ErrorFlags::error()
        } else {
            ErrorFlags::none()
        };
        ival.set_error_flags(flags);
    }
    Ok(())
}

unsafe fn write_outputs(outputs: &[Ival], disc: &RivalDiscretization, out: &[*mut mpfr_t]) {
    for (i, (&out_ptr, val)) in out.iter().zip(outputs).enumerate() {
        if val.error_flags().total() {
            unsafe { mpfr::set_nan(out_ptr) };
        } else {
            let value = disc.convert(i, val.lo());
            unsafe { mpfr::set(out_ptr, value.as_raw(), mpfr::rnd_t::RNDN) };
        }
    }
}

fn write_f64_outputs(outputs: &[Ival], disc: &RivalDiscretization, out: &mut [f64]) {
    for (i, (slot, val)) in out.iter_mut().zip(outputs).enumerate() {
        *slot = if val.error_flags().total() {
            f64::NAN
        } else {
            disc.convert(i, val.lo()).to_f64()
        };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_new(
    builder: *const RivalExprBuilder,
    expr_handles: *const RivalExpr,
    types: *const DiscTypeCode,
    n_exprs: usize,
    options: *const RivalMachineOptions,
    out: *mut *mut RivalMachine,
) -> RivalError {
    status(|| {
        let builder = unsafe { builder.as_ref() }.ok_or(RivalError::Misuse)?;
        let out = unsafe { out.as_mut() }.ok_or(RivalError::Misuse)?;
        let handles = unsafe { c_slice(expr_handles, n_exprs)? };
        let types = unsafe { c_slice(types, n_exprs)? };
        let defaults = RivalMachineOptions::default();
        let options = unsafe { options.as_ref() }.unwrap_or(&defaults);

        let types = types
            .iter()
            .map(|&code| RivalDiscType::try_from(code))
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = builder.outputs(handles).ok_or(RivalError::Misuse)?;

        let n_vars = builder.expressions.argument_count();
        let disc = RivalDiscretization::new(types, options.target_precision());
        let machine = options
            .configure(MachineBuilder::new(disc))
            .build(&builder.expressions, &outputs);
        let instruction_names = InstructionNames::new(&machine.instruction_names());

        let machine = Box::new(RivalMachine {
            machine,
            arg_buf: vec![Ival::zero(f64::MANTISSA_DIGITS); n_vars],
            instruction_names,
            n_vars,
            n_exprs,
        });
        *out = Box::into_raw(machine);
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_free(machine: *mut RivalMachine) {
    if !machine.is_null() {
        unsafe { drop(Box::from_raw(machine)) };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_hints_free(hints: *mut RivalHints) {
    if !hints.is_null() {
        unsafe { drop(Box::from_raw(hints)) };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_instruction_count(machine: *const RivalMachine) -> usize {
    unsafe { machine.as_ref() }.map_or(0, |machine| machine.machine.instruction_count())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_var_count(machine: *const RivalMachine) -> usize {
    unsafe { machine.as_ref() }.map_or(0, |machine| machine.n_vars)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_expr_count(machine: *const RivalMachine) -> usize {
    unsafe { machine.as_ref() }.map_or(0, |machine| machine.n_exprs)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_apply(
    machine: *mut RivalMachine,
    args: *const mpfr_srcptr,
    n_args: usize,
    hints: *const RivalHints,
    max_iterations: u32,
    policy: OutputPolicyCode,
    out: *const mpfr_ptr,
    n_out: usize,
) -> RivalError {
    status(|| {
        let machine = unsafe { machine.as_mut() }.ok_or(RivalError::Misuse)?;
        if n_args != machine.n_vars || n_out != machine.n_exprs {
            return Err(RivalError::Misuse);
        }
        let policy = output_policy(policy)?;
        let hints = unsafe { machine_hints(machine, hints)? };
        let args = unsafe { c_slice(args, n_args)? };
        let out = unsafe { c_slice(out, n_out)? };
        // Check every output before running, so a bad one leaves all of them untouched.
        if out.iter().any(|ptr| ptr.is_null()) {
            return Err(RivalError::Misuse);
        }

        unsafe { marshal_point_args(args, &mut machine.arg_buf)? };
        let outputs =
            machine
                .machine
                .apply(&machine.arg_buf, hints, max_iterations as usize, policy)?;
        unsafe { write_outputs(&outputs, machine.machine.discretization(), out) };
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_apply_f64(
    machine: *mut RivalMachine,
    args: *const f64,
    n_args: usize,
    hints: *const RivalHints,
    max_iterations: u32,
    policy: OutputPolicyCode,
    out: *mut f64,
    n_out: usize,
) -> RivalError {
    status(|| {
        let machine = unsafe { machine.as_mut() }.ok_or(RivalError::Misuse)?;
        if n_args != machine.n_vars || n_out != machine.n_exprs {
            return Err(RivalError::Misuse);
        }
        let policy = output_policy(policy)?;
        let hints = unsafe { machine_hints(machine, hints)? };
        let args = unsafe { c_slice(args, n_args)? };
        let out = unsafe { c_slice_mut(out, n_out)? };

        marshal_f64_args(args, &mut machine.arg_buf);
        let outputs =
            machine
                .machine
                .apply(&machine.arg_buf, hints, max_iterations as usize, policy)?;
        write_f64_outputs(&outputs, machine.machine.discretization(), out);
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_analyze(
    machine: *mut RivalMachine,
    rect: *const mpfr_srcptr,
    n_args: usize,
    hints: *const RivalHints,
    policy: OutputPolicyCode,
    out_analysis: *mut RivalAnalysis,
    out_converged: *mut bool,
    out_hints: *mut *mut RivalHints,
) -> RivalError {
    status(|| {
        let machine = unsafe { machine.as_mut() }.ok_or(RivalError::Misuse)?;
        let out_analysis = unsafe { out_analysis.as_mut() }.ok_or(RivalError::Misuse)?;
        let out_converged = unsafe { out_converged.as_mut() };
        let out_hints = unsafe { out_hints.as_mut() };
        if n_args != machine.n_vars {
            return Err(RivalError::Misuse);
        }
        let policy = output_policy(policy)?;
        let hints = unsafe { machine_hints(machine, hints)? };
        let rect = unsafe { c_slice(rect, 2 * n_args)? };

        unsafe { marshal_rect_args(rect, &mut machine.arg_buf)? };
        let (failure, hints, converged) =
            machine
                .machine
                .analyze_with_hints(&machine.arg_buf, hints, policy);

        *out_analysis = if !failure.lo().is_zero() {
            RivalAnalysis::Invalid
        } else if !failure.hi().is_zero() {
            RivalAnalysis::MaybeInvalid
        } else {
            RivalAnalysis::Valid
        };
        if let Some(out_converged) = out_converged {
            *out_converged = converged;
        }
        if let Some(out_hints) = out_hints {
            *out_hints = Box::into_raw(Box::new(RivalHints { hints }));
        }
        Ok(())
    })
}
