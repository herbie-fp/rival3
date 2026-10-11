mod discretization;
mod expr;
mod machine;
mod options;
mod profile;

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

pub const RIVAL_ABI_VERSION: u32 = 3;

#[unsafe(no_mangle)]
pub extern "C" fn rival_abi_version() -> u32 {
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
