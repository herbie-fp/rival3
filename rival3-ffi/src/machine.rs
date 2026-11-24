use crate::discretization::{RivalDiscType, RivalDiscretization};
use crate::expr::{RivalExpr, RivalExprBuilder};
use crate::options::RivalMachineOptions;
use crate::profile::InstructionNames;
use crate::{
    DiscTypeCode, OutputPolicyCode, RivalError, c_slice, c_slice_mut, mpfr_srcptr, status,
};
use gmp_mpfr_sys::mpfr;
use num_enum::TryFromPrimitive;
use rival::{
    Discretization, Hint, Input, InputRange as CoreInputRange, Ival, Machine, MachineBuilder,
    OutputPolicy,
};
use rug::{Float, Integer, Rational};
use std::ffi::{CStr, c_char};

pub struct RivalMachine {
    pub(crate) machine: Machine<RivalDiscretization>,
    mpfr_outputs: Vec<Float>,
    pub(crate) instruction_names: InstructionNames,
    pub(crate) n_vars: usize,
    pub(crate) n_exprs: usize,
}

pub struct RivalHints {
    hints: Vec<Hint>,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, TryFromPrimitive)]
pub enum RivalInputKind {
    Text = 0,
    F64 = 1,
    Bool = 2,
    I64 = 3,
    U64 = 4,
    Mpfr = 5,
    Rational = 6,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union RivalInputValue {
    pub text: *const c_char,
    pub f64_value: f64,
    pub bool_value: bool,
    pub i64_value: i64,
    pub u64_value: u64,
    pub mpfr_value: mpfr_srcptr,
    pub rational_value: *const RivalRational,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RivalInput {
    pub kind: u32,
    pub value: RivalInputValue,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RivalRational {
    pub numerator: i64,
    pub denominator: u64,
}

// Exact copy of MPFR value
unsafe fn copy_mpfr(value: mpfr_srcptr) -> Result<Float, RivalError> {
    if value.is_null() {
        return Err(RivalError::Misuse);
    }
    let precision = u32::try_from(unsafe { mpfr::get_prec(value) })
        .map_err(|_| RivalError::Misuse)?
        .max(2);
    let mut float = Float::new(precision);
    unsafe { mpfr::set(float.as_raw_mut(), value, mpfr::rnd_t::RNDN) };
    Ok(float)
}

unsafe fn parse_input(input: &RivalInput) -> Result<Input, RivalError> {
    let kind = RivalInputKind::try_from(input.kind)?;
    Ok(match kind {
        RivalInputKind::Text => {
            let text = unsafe { input.value.text };
            if text.is_null() {
                return Err(RivalError::Misuse);
            }
            let text = unsafe { CStr::from_ptr(text) }
                .to_str()
                .map_err(|_| RivalError::Misuse)?;
            Input::Text(text.to_owned())
        }
        RivalInputKind::F64 => Input::F64(unsafe { input.value.f64_value }),
        RivalInputKind::Bool => Input::Bool(unsafe { input.value.bool_value }),
        RivalInputKind::I64 => Input::from(unsafe { input.value.i64_value }),
        RivalInputKind::U64 => Input::from(unsafe { input.value.u64_value }),
        RivalInputKind::Mpfr => Input::Float(unsafe { copy_mpfr(input.value.mpfr_value) }?),
        RivalInputKind::Rational => {
            let value = unsafe { input.value.rational_value.as_ref() }.ok_or(RivalError::Misuse)?;
            if value.denominator == 0 {
                return Err(RivalError::Misuse);
            }
            let numerator = Integer::from(value.numerator);
            let denominator = Integer::from(value.denominator);
            Input::Rational(Rational::from((numerator, denominator)))
        }
    })
}

fn write_analysis(
    failure: &Ival,
    converged: bool,
    hints: Vec<Hint>,
    out_analysis: &mut RivalAnalysis,
    out_converged: Option<&mut bool>,
    out_hints: Option<&mut *mut RivalHints>,
) {
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

fn write_outputs(machine: &mut RivalMachine, outputs: &[Ival], out: &mut [mpfr_srcptr]) {
    machine.mpfr_outputs = outputs
        .iter()
        .map(|val| {
            if val.error_flags().total() {
                Float::with_val(2, rug::float::Special::Nan)
            } else {
                val.lo().clone()
            }
        })
        .collect();
    for (slot, value) in out.iter_mut().zip(&machine.mpfr_outputs) {
        *slot = value.as_raw();
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
            mpfr_outputs: Vec::new(),
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
pub unsafe extern "C" fn rival_machine_apply_mpfr_to_mpfr(
    machine: *mut RivalMachine,
    args: *const mpfr_srcptr,
    n_args: usize,
    hints: *const RivalHints,
    max_iterations: u32,
    policy: OutputPolicyCode,
    out: *mut mpfr_srcptr,
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

        let inputs: Vec<Input> = args
            .iter()
            .map(|&ptr| unsafe { copy_mpfr(ptr).map(Input::from) })
            .collect::<Result<_, _>>()?;
        let outputs = machine
            .machine
            .apply(&inputs, hints, max_iterations as usize, policy)?;
        write_outputs(machine, &outputs, out);
        Ok(())
    })
}

/// Apply typed point values. Text inputs are parsed by Rust at each
/// instruction's working precision.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_apply_any_to_mpfr(
    machine: *mut RivalMachine,
    args: *const RivalInput,
    n_args: usize,
    hints: *const RivalHints,
    max_iterations: u32,
    policy: OutputPolicyCode,
    out: *mut mpfr_srcptr,
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

        let inputs = args
            .iter()
            .map(|input| unsafe { parse_input(input) })
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = machine
            .machine
            .apply(&inputs, hints, max_iterations as usize, policy)?;
        write_outputs(machine, &outputs, out);
        Ok(())
    })
}

/// Apply binary64 inputs and return full-working-precision MPFR endpoints.
/// Output pointers are borrowed from the machine, as with `rival_machine_apply_any_to_mpfr`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_apply_f64_to_mpfr(
    machine: *mut RivalMachine,
    args: *const f64,
    n_args: usize,
    hints: *const RivalHints,
    max_iterations: u32,
    policy: OutputPolicyCode,
    out: *mut mpfr_srcptr,
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

        let inputs: Vec<Input> = args.iter().copied().map(Input::from).collect();
        let outputs = machine
            .machine
            .apply(&inputs, hints, max_iterations as usize, policy)?;
        write_outputs(machine, &outputs, out);
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_apply_f64_to_f64(
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

        let inputs: Vec<Input> = args.iter().copied().map(Input::from).collect();
        let outputs = machine
            .machine
            .apply(&inputs, hints, max_iterations as usize, policy)?;
        write_f64_outputs(&outputs, machine.machine.discretization(), out);
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_analyze_mpfr(
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

        let bounds = rect
            .chunks_exact(2)
            .map(|pair| {
                let (lo_ptr, hi_ptr) = (pair[0], pair[1]);
                if lo_ptr.is_null() || hi_ptr.is_null() {
                    return Err(RivalError::Misuse);
                }
                let lo = unsafe { copy_mpfr(lo_ptr)? };
                let hi = unsafe { copy_mpfr(hi_ptr)? };
                Ok(Input::from(CoreInputRange::new(lo, hi)))
            })
            .collect::<Result<Vec<_>, RivalError>>()?;
        let (failure, hints, converged) =
            machine.machine.analyze_with_hints(&bounds, hints, policy)?;

        write_analysis(
            &failure,
            converged,
            hints,
            out_analysis,
            out_converged,
            out_hints,
        );
        Ok(())
    })
}

/// Analyze typed bounds supplied as `[lo0, hi0, lo1, hi1, ...]`.
/// Text values are parsed as exact decimal or integer-fraction numerals. Other
/// kinds carry native f64, bool, integer, or MPFR values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_analyze(
    machine: *mut RivalMachine,
    bounds: *const RivalInput,
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
        let n_bounds = n_args.checked_mul(2).ok_or(RivalError::Misuse)?;
        let bounds = unsafe { c_slice(bounds, n_bounds)? };
        let bounds = bounds
            .chunks_exact(2)
            .map(|pair| unsafe {
                Ok(Input::from(CoreInputRange::new(
                    parse_input(&pair[0])?,
                    parse_input(&pair[1])?,
                )))
            })
            .collect::<Result<Vec<_>, RivalError>>()?;
        let (failure, hints, converged) =
            machine.machine.analyze_with_hints(&bounds, hints, policy)?;
        write_analysis(
            &failure,
            converged,
            hints,
            out_analysis,
            out_converged,
            out_hints,
        );
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_analyze_f64(
    machine: *mut RivalMachine,
    bounds: *const f64,
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
        let n_bounds = n_args.checked_mul(2).ok_or(RivalError::Misuse)?;
        let bounds = unsafe { c_slice(bounds, n_bounds)? };
        let bounds: Vec<Input> = bounds
            .chunks_exact(2)
            .map(|pair| Input::from(CoreInputRange::new(pair[0], pair[1])))
            .collect();
        let (failure, hints, converged) =
            machine.machine.analyze_with_hints(&bounds, hints, policy)?;
        write_analysis(
            &failure,
            converged,
            hints,
            out_analysis,
            out_converged,
            out_hints,
        );
        Ok(())
    })
}
