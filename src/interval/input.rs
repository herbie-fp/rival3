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
                Ok(enclose(precision, &self))
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
        parse(precision, &self)
    }
}

impl IntoIval for &String {
    fn into_ival(self, precision: u32) -> Result<Ival, ParseIvalError> {
        parse(precision, self)
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

fn enclose<T: Clone>(precision: u32, value: T) -> Ival
where
    Float: AssignRound<T, Round = Round, Ordering = Ordering>,
{
    let lo = Float::with_val_round(precision, value.clone(), Round::Down);
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
        let numerator =
            Integer::from(Integer::parse(numerator.trim()).map_err(|_| ParseIvalError)?);
        let denominator =
            Integer::from(Integer::parse(denominator.trim()).map_err(|_| ParseIvalError)?);
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

#[cfg(test)]
mod tests {
    use super::*;
    use rug::ops::Pow;

    #[test]
    fn rational_enclosures_refine() {
        for value in [Rational::from((1, 3)), Rational::from((-1, 3))] {
            let coarse = Ival::with_val(8, &value).unwrap();
            let fine = Ival::with_val(100, &value).unwrap();
            assert!(coarse.lo() < fine.lo());
            assert!(fine.lo() < &value);
            assert!(&value < fine.hi());
            assert!(fine.hi() < coarse.hi());
            assert!(!fine.lo_immovable() && !fine.hi_immovable());
        }
        let exact = Ival::with_val(8, Rational::from((1, 8))).unwrap();
        assert_eq!(exact.lo(), exact.hi());
        assert!(exact.lo_immovable() && exact.hi_immovable());
    }

    #[test]
    fn strings_preserve_exact_decimal_values() {
        for (text, value) in [
            ("0.1", Rational::from((1, 10))),
            ("-7/3", Rational::from((-7, 3))),
            ("1e330", Rational::from(Integer::from(10).pow(330))),
            (
                "1e-330",
                Rational::from((Integer::from(1), Integer::from(10).pow(330))),
            ),
        ] {
            let expected = Ival::with_val(80, &value).unwrap();
            assert_eq!(Ival::with_val(80, text).unwrap(), expected);
            let owned = text.to_owned();
            assert_eq!(Ival::with_val(80, &owned).unwrap(), expected);
            assert_eq!(Ival::with_val(80, owned).unwrap(), expected);
        }
        assert_ne!(
            Ival::with_val(100, "0.1").unwrap(),
            Ival::with_val(100, 0.1f64).unwrap()
        );
        for text in ["", "bad", "1/0", "1/2/3", "1.2/3"] {
            assert!(Ival::with_val(53, text).is_err(), "{text}");
            assert!(Ival::with_val(53, text.to_owned()).is_err(), "{text}");
        }
    }

    #[test]
    fn floats_mpfr_integers_and_booleans() {
        let float = Float::with_val(100, Rational::from((1, 3)));
        let interval = Ival::with_val(100, &float).unwrap();
        assert_eq!(interval.lo(), &float);
        assert_eq!(interval.hi(), &float);
        assert!(interval.lo_immovable() && interval.hi_immovable());
        let coarse = Ival::with_val(8, &float).unwrap();
        assert!(coarse.lo() < &float && coarse.hi() > &float);
        assert!(!coarse.lo_immovable() && !coarse.hi_immovable());
        let integer = (Integer::from(1) << 200u32) + 1;
        let interval = Ival::with_val(201, &integer).unwrap();
        assert_eq!(interval.lo(), &integer);
        assert_eq!(interval.hi(), &integer);
        for value in [false, true] {
            assert_eq!(
                Ival::with_val(2, value).unwrap(),
                Ival::bool_interval(value, value)
            );
        }
        let zero = Ival::with_val(53, -0.0f64).unwrap();
        assert!(zero.lo().is_sign_negative() && zero.hi().is_sign_negative());
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(Ival::with_val(53, value).unwrap().error_flags().total());
        }
    }
}
