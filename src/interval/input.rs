//! Construct enclosures of scalar inputs without intermediate rounding.

use std::cmp::Ordering;

use rug::{Float, Integer, Rational, float::Round, ops::AssignRound};

use super::Ival;

/// A string could not be parsed as a number or an integer fraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("expected a number, boolean, or integer fraction with a nonzero denominator")]
pub struct ParseIvalError;

/// A scalar value that can be enclosed at a requested precision.
///
/// Used by [`Ival::with_val`]. Numeric implementations cannot fail to parse;
/// strings return [`ParseIvalError`] for malformed values.
pub trait IntoIval {
    fn into_ival(self, precision: u32) -> Result<Ival, ParseIvalError>;
}

macro_rules! impl_numeric_input {
    ($($ty:ty),+ $(,)?) => {$(
        impl IntoIval for $ty {
            fn into_ival(self, precision: u32) -> Result<Ival, ParseIvalError> {
                (&self).into_ival(precision)
            }
        }

        impl IntoIval for &$ty {
            fn into_ival(self, precision: u32) -> Result<Ival, ParseIvalError> {
                Ok(enclose(precision, self))
            }
        }
    )+};
}

impl_numeric_input!(
    bool, f32, f64, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, Integer,
    Rational, Float,
);

impl IntoIval for &str {
    fn into_ival(self, precision: u32) -> Result<Ival, ParseIvalError> {
        parse(precision, self)
    }
}

impl IntoIval for String {
    fn into_ival(self, precision: u32) -> Result<Ival, ParseIvalError> {
        self.as_str().into_ival(precision)
    }
}

impl IntoIval for &String {
    fn into_ival(self, precision: u32) -> Result<Ival, ParseIvalError> {
        self.as_str().into_ival(precision)
    }
}

impl Ival {
    /// Enclose a scalar value at `precision` bits, rounding outwards.
    ///
    /// Accepts primitive numbers and booleans, and owned or borrowed Rug
    /// integers, rationals, MPFR floats, and strings. A float supplies its existing
    /// binary value; no decimal interpretation or intermediate rounding occurs.
    /// Only exactly represented endpoints are marked immovable. Nonfinite
    /// floating-point inputs produce error intervals, as with [`Self::from_lo_hi`].
    ///
    /// Strings may contain decimal/scientific notation (using Rug's decimal
    /// float parser), integer fractions such as `"1/3"`, or `"true"`/`"false"`.
    /// Fractions require a nonzero denominator. Nonfinite spellings accepted
    /// by Rug produce error intervals, just like nonfinite typed floats.
    /// Malformed strings return [`ParseIvalError`]; typed numeric inputs return
    /// `Ok`. All inputs use the same `Result` return type.
    ///
    /// The returned interval does not retain the source value. To refine it,
    /// call this constructor again with the original value and higher precision.
    /// Panics if `precision` is outside MPFR's supported range.
    ///
    /// ```
    /// use rival::Ival;
    /// use rug::Rational;
    /// let value = Rational::from((1, 3));
    /// let interval = Ival::with_val(53, &value).unwrap();
    /// assert!(interval.lo() < &value);
    /// assert!(interval.hi() > &value);
    ///
    /// assert_eq!(interval, Ival::with_val(53, "1/3").unwrap());
    /// let interval = Ival::with_val(53, "1e330").unwrap();
    /// assert!(interval.lo().is_finite());
    /// assert!(interval.hi().is_finite());
    /// ```
    pub fn with_val<T: IntoIval>(precision: u32, value: T) -> Result<Self, ParseIvalError> {
        value.into_ival(precision)
    }

    fn from_rounded((lo, lo_round): (Float, Ordering), (hi, hi_round): (Float, Ordering)) -> Self {
        let mut interval = Self::from_lo_hi(lo, hi);
        interval.set_immovable(lo_round == Ordering::Equal, hi_round == Ordering::Equal);
        interval
    }
}

fn enclose<'a, T>(precision: u32, value: &'a T) -> Ival
where
    Float: AssignRound<&'a T, Round = Round, Ordering = Ordering>,
{
    let lo = Float::with_val_round(precision, value, Round::Down);
    let hi = Float::with_val_round(precision, value, Round::Up);
    Ival::from_rounded(lo, hi)
}

fn parse(precision: u32, text: &str) -> Result<Ival, ParseIvalError> {
    let text = text.trim();
    match text {
        "true" => return Ival::with_val(precision, true),
        "false" => return Ival::with_val(precision, false),
        _ => {}
    }
    if let Some((numerator, denominator)) = text.split_once('/') {
        let numerator = numerator
            .trim()
            .parse::<Integer>()
            .map_err(|_| ParseIvalError)?;
        let denominator = denominator
            .trim()
            .parse::<Integer>()
            .map_err(|_| ParseIvalError)?;
        if denominator == 0 {
            return Err(ParseIvalError);
        }
        return Ival::with_val(precision, Rational::from((numerator, denominator)));
    }
    // Rug's deferred parse value is consumed when rounded, so parse each
    // endpoint independently rather than rounding a rounded Float again.
    let lo = Float::with_val_round(
        precision,
        Float::parse(text).map_err(|_| ParseIvalError)?,
        Round::Down,
    );
    let hi = Float::with_val_round(
        precision,
        Float::parse(text).map_err(|_| ParseIvalError)?,
        Round::Up,
    );
    Ok(Ival::from_rounded(lo, hi))
}
