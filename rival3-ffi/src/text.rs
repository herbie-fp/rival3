//! Tagged text transport for language bindings, retaining exact input values.

use crate::discretization::{RivalDiscType, truncated_integer};
use crate::machine::{RivalHints, RivalMachine, machine_hints, output_policy};
use crate::{OutputPolicyCode, RivalError, c_slice, c_slice_mut, status};
use rival::{Discretization, Ival};
use rug::{Float, Integer, Rational, float::Round};
use std::cmp::Ordering;
use std::ffi::{CStr, CString, c_char};
use std::ptr;

enum Input {
    Exact(Ival),
    Rational(Rational),
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RivalInputKind {
    Integer = 0,
    Float = 1,
    Rational = 2,
    Boolean = 3,
}

#[repr(C)]
pub struct RivalTextInput {
    pub kind: u32,
    pub value: *const c_char,
}

fn integer(text: &str) -> Result<Integer, RivalError> {
    Integer::parse(text)
        .map(Integer::from)
        .map_err(|_| RivalError::Misuse)
}

fn point(value: Float) -> Ival {
    let mut interval = Ival::from_lo_hi(value.clone(), value);
    interval.set_immovable(true, true);
    interval
}

impl Input {
    fn parse(kind: u32, text: &str) -> Result<Self, RivalError> {
        Ok(match kind {
            x if x == RivalInputKind::Integer as u32 => {
                let value = text;
                let value = integer(value)?;
                Self::Exact(point(Float::with_val(
                    value.significant_bits().max(2),
                    value,
                )))
            }
            // Parse as f64 before MPFR so Python floats keep their binary value
            // and signed zero. Nonfinite floats become invalid point inputs.
            x if x == RivalInputKind::Float as u32 => Self::Exact(point(Float::with_val(
                53,
                text.parse::<f64>().map_err(|_| RivalError::Misuse)?,
            ))),
            x if x == RivalInputKind::Boolean as u32 => match text {
                "0" => Self::Exact(Ival::bool_interval(false, false)),
                "1" => Self::Exact(Ival::bool_interval(true, true)),
                _ => return Err(RivalError::Misuse),
            },
            x if x == RivalInputKind::Rational as u32 => {
                let (num, den) = text.split_once('/').ok_or(RivalError::Misuse)?;
                let (num, den) = (integer(num)?, integer(den)?);
                if den == 0 {
                    return Err(RivalError::Misuse);
                }
                Self::Rational(Rational::from((num, den)))
            }
            _ => return Err(RivalError::Misuse),
        })
    }

    fn enclose(&self, precision: u32) -> Ival {
        match self {
            Self::Exact(value) => value.clone(),
            Self::Rational(value) => {
                let (lo, lo_round) = Float::with_val_round(precision, value, Round::Down);
                let (hi, hi_round) = Float::with_val_round(precision, value, Round::Up);
                let mut interval = Ival::from_lo_hi(lo, hi);
                interval.set_immovable(lo_round == Ordering::Equal, hi_round == Ordering::Equal);
                interval
            }
        }
    }
}

/// Evaluate text inputs with separate kind tags. Values are decimal integers,
/// f64 representations, `0`/`1` booleans, or rational numerator/denominator.
/// Rational bounds are refined at increasing precision when necessary.
/// Every declared argument must have a non-null, NUL-terminated input string.
///
/// On success, each output is newly allocated decimal text, or NULL for an
/// invalid output under AllowPartial. Bool outputs are "0" or "1". Integer
/// outputs truncate toward zero. Free each non-null output with
/// `rival_string_free`. Output slots are untouched on error. Do not pass slots
/// that still own strings from a previous call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_machine_apply_text(
    machine: *mut RivalMachine,
    args: *const RivalTextInput,
    n_args: usize,
    hints: *const RivalHints,
    max_iterations: u32,
    policy: OutputPolicyCode,
    out: *mut *mut c_char,
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
            .map(|arg| {
                if arg.value.is_null() {
                    return Err(RivalError::Misuse);
                }
                let text = unsafe { CStr::from_ptr(arg.value) }
                    .to_str()
                    .map_err(|_| RivalError::Misuse)?;
                Input::parse(arg.kind, text)
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (arg, input) in machine.arg_buf.iter_mut().zip(&inputs) {
            *arg = input.enclose(machine.machine.argument_precision());
        }
        let refine = machine
            .arg_buf
            .iter()
            .any(|arg| !arg.lo_immovable() || !arg.hi_immovable());
        let outputs = if refine {
            machine.machine.apply_refining(
                &machine.arg_buf,
                hints,
                max_iterations as usize,
                policy,
                |precision, args| {
                    for (arg, input) in args.iter_mut().zip(&inputs) {
                        *arg = input.enclose(precision);
                    }
                },
            )?
        } else {
            machine
                .machine
                .apply(&machine.arg_buf, hints, max_iterations as usize, policy)?
        };
        let disc = machine.machine.discretization();
        // Allocate all strings before publishing any pointers, including when
        // an integer conversion cannot produce a finite result.
        let strings = outputs
            .iter()
            .enumerate()
            .map(|(i, value)| {
                if value.error_flags().total() {
                    return Ok(None);
                }
                let text = match disc.type_at(i) {
                    RivalDiscType::Int => truncated_integer(value.lo())
                        .ok_or(RivalError::Unsamplable)?
                        .to_string(),
                    RivalDiscType::Bool => if value.lo().is_zero() { "0" } else { "1" }.to_owned(),
                    RivalDiscType::F32 | RivalDiscType::F64 => {
                        disc.convert(i, value.lo()).to_f64().to_string()
                    }
                };
                Ok(Some(CString::new(text).map_err(|_| RivalError::Misuse)?))
            })
            .collect::<Result<Vec<_>, RivalError>>()?;
        for (slot, value) in out.iter_mut().zip(strings) {
            *slot = value.map_or(ptr::null_mut(), CString::into_raw);
        }
        Ok(())
    })
}

/// Free one string returned by `rival_machine_apply_text`. NULL is allowed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_string_free(value: *mut c_char) {
    if !value.is_null() {
        unsafe { drop(CString::from_raw(value)) };
    }
}
