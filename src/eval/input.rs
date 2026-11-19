//! Original runtime values retained while input instructions are reevaluated.

use rug::{Float, Integer, Rational};

use crate::{ErrorFlags, IntoIval, Ival, ParseIvalError};

/// A runtime input whose enclosure can be recomputed at a requested precision.
///
/// Strings retain their exact spelling, rationals retain their exact value,
/// and floating-point inputs retain their existing binary value. An interval
/// is already an enclosure and is copied without changing its bounds or flags.
#[derive(Clone, Debug)]
pub enum Input {
    Bool(bool),
    F64(f64),
    Float(Float),
    Integer(Integer),
    Rational(Rational),
    Text(String),
    Interval(Ival),
}

/// A runtime interval specified by its lower and upper values.
#[derive(Clone, Debug)]
pub struct InputRange {
    pub lower: Input,
    pub upper: Input,
}

impl InputRange {
    pub fn new(lower: impl Into<Input>, upper: impl Into<Input>) -> Self {
        Self {
            lower: lower.into(),
            upper: upper.into(),
        }
    }

    pub(crate) fn to_ival(&self, precision: u32) -> Result<Ival, ParseIvalError> {
        let lower = Ival::with_val(precision, &self.lower)?;
        let upper = Ival::with_val(precision, &self.upper)?;
        let mut interval = Ival::from_lo_hi(lower.lo().clone(), upper.hi().clone());
        interval.set_immovable(lower.lo_immovable(), upper.hi_immovable());
        let lower_err = lower.error_flags();
        let upper_err = upper.error_flags();
        interval.set_error_flags(ErrorFlags::new(
            lower_err.partial() || upper_err.partial(),
            lower_err.total() || upper_err.total(),
        ));
        Ok(interval)
    }
}

impl IntoIval for &Input {
    fn into_ival(self, precision: u32) -> Result<Ival, ParseIvalError> {
        match self {
            Input::Bool(value) => Ival::with_val(precision, value),
            Input::F64(value) => Ival::with_val(precision, value),
            Input::Float(value) => Ival::with_val(precision, value),
            Input::Integer(value) => Ival::with_val(precision, value),
            Input::Rational(value) => Ival::with_val(precision, value),
            Input::Text(value) => Ival::with_val(precision, value),
            Input::Interval(value) => Ok(value.clone()),
        }
    }
}

impl IntoIval for Input {
    fn into_ival(self, precision: u32) -> Result<Ival, ParseIvalError> {
        (&self).into_ival(precision)
    }
}

macro_rules! from_input {
    ($($ty:ty => $variant:ident),+ $(,)?) => {$(
        impl From<$ty> for Input {
            fn from(value: $ty) -> Self { Self::$variant(value) }
        }
    )+};
}
from_input!(bool => Bool, f64 => F64, Float => Float, Integer => Integer,
            Rational => Rational, String => Text, Ival => Interval);

impl From<&str> for Input {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

impl From<f32> for Input {
    fn from(value: f32) -> Self {
        Self::F64(f64::from(value))
    }
}

macro_rules! from_integer {
    ($($ty:ty),+ $(,)?) => {$(
        impl From<$ty> for Input {
            fn from(value: $ty) -> Self { Self::Integer(Integer::from(value)) }
        }
    )+};
}
from_integer!(
    i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize
);
