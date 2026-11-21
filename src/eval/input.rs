//! Original runtime values retained while input instructions are reevaluated.

use rug::{Float, Integer, Rational};

use crate::{ErrorFlags, IntoIval, Ival, ParseIvalError};

/// A scalar runtime input whose enclosure can be recomputed at a requested precision.
///
/// Strings retain their exact spelling, rationals retain their exact value,
/// and floating-point inputs retain their existing binary value.
#[derive(Clone, Debug)]
pub enum Input {
    Bool(bool),
    F64(f64),
    Float(Float),
    Integer(Integer),
    Rational(Rational),
    Text(String),
    Range(Box<InputRange>),
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
        if lower.lo() > upper.hi() {
            return Err(ParseIvalError);
        }
        let mut interval = Ival::from_lo_hi(lower.lo().clone(), upper.hi().clone());
        // A non-singleton input range has intentional width. Its exact bounds
        // must not make that width look like a precision-stuck result.
        let singleton = lower.lo() == upper.hi() && lower.lo_immovable() && upper.hi_immovable();
        interval.set_immovable(singleton, singleton);
        let lower_err = lower.error_flags();
        let upper_err = upper.error_flags();
        // Infinite endpoints are valid bounds when they face outwards.
        let lower_is_neg_infinity = lower.lo().is_infinite() && lower.lo().is_sign_negative();
        let upper_is_pos_infinity = upper.hi().is_infinite() && upper.hi().is_sign_positive();
        let lower_invalid = lower_err.total() && !lower_is_neg_infinity;
        let upper_invalid = upper_err.total() && !upper_is_pos_infinity;
        interval.set_error_flags(ErrorFlags::new(
            (lower_err.partial() && !lower_is_neg_infinity)
                || (upper_err.partial() && !upper_is_pos_infinity),
            lower_invalid || upper_invalid,
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
            Input::Range(value) => value.to_ival(precision),
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
            Rational => Rational, String => Text);

impl From<InputRange> for Input {
    fn from(value: InputRange) -> Self {
        Self::Range(Box::new(value))
    }
}

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

impl Input {
    pub(crate) fn is_range(&self) -> bool {
        matches!(self, Self::Range(_))
    }
}
