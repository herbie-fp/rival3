mod discretization;
mod expr;
mod machine;
mod options;
mod profile;

use gmp_mpfr_sys::mpfr::mpfr_t;
use num_enum::{TryFromPrimitive, TryFromPrimitiveError};
use rival::RivalError as CoreError;
use std::slice;

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RivalError {
    Ok = 0,
    InvalidInput = -1,
    Unsamplable = -2,
    Misuse = -3,
}

impl From<CoreError> for RivalError {
    fn from(error: CoreError) -> Self {
        match error {
            CoreError::InvalidInput => Self::InvalidInput,
            CoreError::Unsamplable => Self::Unsamplable,
        }
    }
}

impl<T: TryFromPrimitive> From<TryFromPrimitiveError<T>> for RivalError {
    fn from(_: TryFromPrimitiveError<T>) -> Self {
        Self::Misuse
    }
}

pub type UnaryOpCode = u32;
pub type UnaryParamOpCode = u32;
pub type BinaryOpCode = u32;
pub type TernaryOpCode = u32;
pub type DiscTypeCode = u32;
pub type StrategyCode = u32;
pub type OutputPolicyCode = u32;

// MPFR's own pointer types.
#[allow(non_camel_case_types)]
pub type mpfr_srcptr = *const mpfr_t;
#[allow(non_camel_case_types)]
pub type mpfr_ptr = *mut mpfr_t;

pub const RIVAL_ABI_VERSION: u32 = 3;

#[unsafe(no_mangle)]
pub extern "C" fn rival_version() -> u32 {
    RIVAL_ABI_VERSION
}

/// Run a body that uses `?` and return its result as a status code.
fn status(body: impl FnOnce() -> Result<(), RivalError>) -> RivalError {
    body().err().unwrap_or(RivalError::Ok)
}

/// Borrow a C array, which may be NULL only when `len` is zero.
unsafe fn c_slice<'a, T>(ptr: *const T, len: usize) -> Result<&'a [T], RivalError> {
    match len {
        0 => Ok(&[]),
        _ if ptr.is_null() => Err(RivalError::Misuse),
        _ => Ok(unsafe { slice::from_raw_parts(ptr, len) }),
    }
}

/// Mutably borrow a C array, which may be NULL only when `len` is zero.
unsafe fn c_slice_mut<'a, T>(ptr: *mut T, len: usize) -> Result<&'a mut [T], RivalError> {
    match len {
        0 => Ok(&mut []),
        _ if ptr.is_null() => Err(RivalError::Misuse),
        _ => Ok(unsafe { slice::from_raw_parts_mut(ptr, len) }),
    }
}
