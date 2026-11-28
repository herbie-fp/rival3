//! Transcendental functions computed by Arb, rounded as MPFR rounds them.
use std::ffi::c_int;
use std::mem::MaybeUninit;

pub(crate) use flint3_sys::{
    arb_acos, arb_acosh, arb_asin, arb_asinh, arb_atan, arb_atan2, arb_atanh, arb_cos, arb_cos_pi,
    arb_cosh, arb_exp, arb_hypgeom_erf, arb_hypgeom_erfc, arb_lgamma, arb_log, arb_log1p, arb_pow,
    arb_sin, arb_sin_pi, arb_sinh, arb_tan, arb_tan_pi,
};
use flint3_sys::{
    arb_clear, arb_div_ui, arb_init, arb_is_finite, arb_log_base_ui, arb_mul_2exp_si, arb_set_ui,
    arb_struct, arf_abs_bound_lt_2exp_si, arf_struct, slong,
};
use gmp_mpfr_sys::mpfr::{self, mpfr_t, rnd_t};
use rug::{Float, float::Round};

use crate::mpfr::{mpfr_get_exp, to_mpfr_round};

// FLINT declares these after `mpfr.h`, so its Rust bindings omit them.
unsafe extern "C" {
    fn arf_set_mpfr(x: *mut arf_struct, y: *const mpfr_t);
    fn arf_get_mpfr(y: *mut mpfr_t, x: *const arf_struct, rnd: rnd_t) -> c_int;
    fn arb_can_round_mpfr(x: *const arb_struct, prec: slong, rnd: rnd_t) -> c_int;
}

const GUARD_BITS: slong = 32;

pub(crate) type ArbUnary = unsafe extern "C" fn(*mut arb_struct, *const arb_struct, slong);
pub(crate) type ArbBinary =
    unsafe extern "C" fn(*mut arb_struct, *const arb_struct, *const arb_struct, slong);

struct Arb(arb_struct);

impl Arb {
    fn new() -> Self {
        let mut ball = MaybeUninit::uninit();
        unsafe {
            arb_init(ball.as_mut_ptr());
            Arb(ball.assume_init())
        }
    }

    // MPFR's zeros, infinities and NaN follow conventions Arb lacks.
    fn exact(x: &Float) -> Option<Self> {
        x.is_normal().then(|| {
            let mut ball = Arb::new();
            unsafe { arf_set_mpfr(&mut ball.0.mid, x.as_raw()) };
            ball
        })
    }

    // MPFR decides the sign of a zero and what overflows and underflows
    // round to.
    fn is_regular(&self) -> bool {
        unsafe {
            let exponents = mpfr::get_emin() as slong..=mpfr::get_emax() as slong;
            arb_is_finite(&self.0) != 0
                && exponents.contains(&arf_abs_bound_lt_2exp_si(&self.0.mid))
        }
    }
}

impl Drop for Arb {
    fn drop(&mut self) {
        unsafe { arb_clear(&mut self.0) }
    }
}

fn rounded(
    out: &mut Float,
    rnd: Round,
    operand_bits: u32,
    eval: impl Fn(*mut arb_struct, slong),
) -> Option<bool> {
    let evaluate = |prec: slong| {
        let mut ball = Arb::new();
        eval(&mut ball.0, prec);
        ball
    };
    let prec = out.prec() as slong;
    let rnd = to_mpfr_round(rnd);
    // Overflow and underflow show at any precision.
    if !evaluate(GUARD_BITS).is_regular() {
        return None;
    }
    // A truncated series in short operands can be representable. Its next
    // term is under twice as deep, or as deep as the operands.
    let precise = prec + GUARD_BITS;
    let deep = (2 * precise).max(operand_bits as slong + GUARD_BITS);
    for working in [precise, deep] {
        let ball = evaluate(working);
        unsafe {
            if !ball.is_regular() {
                return None;
            }
            if arb_can_round_mpfr(&ball.0, prec, rnd) != 0 {
                return Some(arf_get_mpfr(out.as_raw_mut(), &ball.0.mid, rnd) == 0);
            }
        }
    }
    None
}

pub(crate) fn unary(f: ArbUnary, x: &Float, out: &mut Float, rnd: Round) -> Option<bool> {
    let exact = Arb::exact(x)?;
    rounded(out, rnd, x.prec(), |ball, prec| unsafe {
        f(ball, &exact.0, prec)
    })
}

pub(crate) fn binary(
    f: ArbBinary,
    x: &Float,
    y: &Float,
    out: &mut Float,
    rnd: Round,
) -> Option<bool> {
    let (exact_x, exact_y) = (Arb::exact(x)?, Arb::exact(y)?);
    rounded(out, rnd, x.prec().max(y.prec()), |ball, prec| unsafe {
        f(ball, &exact_x.0, &exact_y.0, prec)
    })
}

// `f` takes a multiple of pi, so this computes `f(2x / n)`.
pub(crate) fn periodic(
    f: ArbUnary,
    x: &Float,
    n: u64,
    out: &mut Float,
    rnd: Round,
) -> Option<bool> {
    let exact = Arb::exact(x)?;
    // Only the quotient modulo 2 matters, so its integer bits are extra.
    let integer_bits = mpfr_get_exp(x).max(0) as slong;
    rounded(out, rnd, x.prec(), |ball, prec| unsafe {
        arb_mul_2exp_si(ball, &exact.0, 1);
        arb_div_ui(ball, ball, n, prec + integer_bits);
        f(ball, ball, prec);
    })
}

pub(crate) unsafe extern "C" fn arb_exp2(y: *mut arb_struct, x: *const arb_struct, prec: slong) {
    unsafe {
        arb_set_ui(y, 2);
        arb_pow(y, y, x, prec);
    }
}

pub(crate) unsafe extern "C" fn arb_log2(y: *mut arb_struct, x: *const arb_struct, prec: slong) {
    unsafe { arb_log_base_ui(y, x, 2, prec) }
}

pub(crate) unsafe extern "C" fn arb_log10(y: *mut arb_struct, x: *const arb_struct, prec: slong) {
    unsafe { arb_log_base_ui(y, x, 10, prec) }
}
