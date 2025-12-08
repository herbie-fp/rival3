use super::value::{Endpoint, Ival, IvalClass, classify};
use crate::{
    interval::core::endpoint_unary,
    mpfr::{
        mpfr_cos, mpfr_cosu, mpfr_div, mpfr_floor_inplace, mpfr_get_exp, mpfr_pi,
        mpfr_round_inplace, mpfr_sin, mpfr_sinu, mpfr_tan, mpfr_tanu, zero,
    },
};
use rug::{Assign, Float, float::Round};

const RANGE_REDUCE_PRECISION_CAP: u32 = 1 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PeriodClass {
    TooWide,
    NearZero,
    RangeReduce,
}

fn classify_ival_periodic(x: &Ival, period_quarter_bitlen: i64) -> PeriodClass {
    let (xlo, xhi) = (x.lo.as_float(), x.hi.as_float());

    if xlo.is_infinite() || xhi.is_infinite() {
        return PeriodClass::TooWide;
    }

    let (lo_exp, hi_exp) = (mpfr_get_exp(xlo), mpfr_get_exp(xhi));
    if lo_exp < period_quarter_bitlen && hi_exp < period_quarter_bitlen {
        return PeriodClass::NearZero;
    }

    let lo_ulp = lo_exp.saturating_sub(xlo.prec() as i64);
    let hi_ulp = hi_exp.saturating_sub(xhi.prec() as i64);

    if lo_ulp > 0 || hi_ulp > 0 {
        if xlo == xhi {
            PeriodClass::RangeReduce
        } else {
            PeriodClass::TooWide
        }
    } else {
        PeriodClass::RangeReduce
    }
}

fn range_reduce_precision(xlo: &Float, xhi: &Float, curr_prec: u32) -> u32 {
    let lo = (mpfr_get_exp(xlo) + 2 * (xlo.prec() as i64)).max(curr_prec as i64) as u32;
    let hi = (mpfr_get_exp(xhi) + 2 * (xhi.prec() as i64)).max(curr_prec as i64) as u32;
    lo.max(hi).min(RANGE_REDUCE_PRECISION_CAP).max(curr_prec)
}

fn ival_div_pi(x: &Ival, prec: u32, round_fn: fn(&mut Float) -> bool) -> (Float, Float) {
    let (mut pi_lo, mut pi_hi) = (zero(prec), zero(prec));
    mpfr_pi(&mut pi_lo, Round::Down);
    mpfr_pi(&mut pi_hi, Round::Up);

    let (mut q_lo, mut q_hi) = (zero(prec), zero(prec));
    mpfr_div(x.lo.as_float(), &pi_hi, &mut q_lo, Round::Down);
    mpfr_div(x.hi.as_float(), &pi_lo, &mut q_hi, Round::Up);
    round_fn(&mut q_lo);
    round_fn(&mut q_hi);
    (q_lo, q_hi)
}

fn ival_div_n_half(
    x: &Ival,
    n: u64,
    prec: u32,
    round_fn: fn(&mut Float) -> bool,
) -> (Float, Float) {
    let n_half = Float::with_val(prec, n) / 2u32;

    let (mut q_lo, mut q_hi) = (zero(prec), zero(prec));
    mpfr_div(x.lo.as_float(), &n_half, &mut q_lo, Round::Down);
    mpfr_div(x.hi.as_float(), &n_half, &mut q_hi, Round::Up);
    round_fn(&mut q_lo);
    round_fn(&mut q_hi);
    (q_lo, q_hi)
}

fn bfeven(x: &Float) -> bool {
    let prec = x.prec();
    let mut t = Float::with_val(prec, x);
    t /= 2;
    mpfr_floor_inplace(&mut t);
    t *= 2;
    t == *x
}

#[inline]
fn bfodd(x: &Float) -> bool {
    !bfeven(x)
}

fn bfsub_is_one(a: &Float, b: &Float) -> bool {
    let prec = a.prec().max(b.prec());
    let mut d = Float::with_val(prec, b);
    d -= a;
    d == 1
}

fn period_quarter_bitlen(n: u64, divisor: u64) -> i64 {
    let quarter = n / divisor;
    if quarter > 0 {
        (quarter.ilog2() + 1) as i64
    } else {
        0
    }
}

fn endpoint_min<F>(f: &F, lo: &Endpoint, hi: &Endpoint, out: &mut Float) -> bool
where
    F: Fn(&Float, &mut Float, Round) -> bool,
{
    let prec = out.prec();
    let mut tmp = zero(prec);

    let imm_lo = endpoint_unary(f, lo, out, Round::Down);
    let imm_hi = endpoint_unary(f, hi, &mut tmp, Round::Down);

    if tmp < *out {
        out.assign(&tmp);
        imm_hi
    } else if tmp == *out {
        imm_lo || imm_hi
    } else {
        imm_lo
    }
}

fn endpoint_max<F>(f: &F, lo: &Endpoint, hi: &Endpoint, out: &mut Float) -> bool
where
    F: Fn(&Float, &mut Float, Round) -> bool,
{
    let prec = out.prec();
    let mut tmp = zero(prec);

    let imm_lo = endpoint_unary(f, lo, out, Round::Up);
    let imm_hi = endpoint_unary(f, hi, &mut tmp, Round::Up);

    if tmp > *out {
        out.assign(&tmp);
        imm_hi
    } else if tmp == *out {
        imm_lo || imm_hi
    } else {
        imm_lo
    }
}

impl Ival {
    pub fn cos_assign(&mut self, x: &Ival) {
        self.err = x.err;
        let (xlo, xhi) = (x.lo.as_float(), x.hi.as_float());

        match classify_ival_periodic(x, 1) {
            PeriodClass::TooWide => {
                self.lo.as_float_mut().assign(-1);
                self.hi.as_float_mut().assign(1);
                self.lo.immovable = false;
                self.hi.immovable = false;
            }
            PeriodClass::NearZero => match classify(x, false) {
                IvalClass::Neg => self.monotonic_assign(&mpfr_cos, x),
                IvalClass::Pos => self.comonotonic_assign(&mpfr_cos, x),
                IvalClass::Mix => {
                    self.set_prec(x.prec());
                    self.lo.immovable =
                        endpoint_min(&mpfr_cos, &x.lo, &x.hi, self.lo.as_float_mut());
                    self.hi.as_float_mut().assign(1);
                    self.hi.immovable = false;
                }
            },
            PeriodClass::RangeReduce => {
                let prec = range_reduce_precision(xlo, xhi, x.prec());
                let (a, b) = ival_div_pi(x, prec, mpfr_floor_inplace);

                if a == b && bfeven(&a) {
                    self.comonotonic_assign(&mpfr_cos, x);
                } else if a == b && bfodd(&a) {
                    self.monotonic_assign(&mpfr_cos, x);
                } else if bfsub_is_one(&a, &b) && bfeven(&a) {
                    self.set_prec(x.prec());
                    self.lo.as_float_mut().assign(-1);
                    self.lo.immovable = false;
                    self.hi.immovable =
                        endpoint_max(&mpfr_cos, &x.lo, &x.hi, self.hi.as_float_mut());
                } else if bfsub_is_one(&a, &b) && bfodd(&a) {
                    self.set_prec(x.prec());
                    self.lo.immovable =
                        endpoint_min(&mpfr_cos, &x.lo, &x.hi, self.lo.as_float_mut());
                    self.hi.as_float_mut().assign(1);
                    self.hi.immovable = false;
                } else {
                    self.lo.as_float_mut().assign(-1);
                    self.hi.as_float_mut().assign(1);
                    self.lo.immovable = false;
                    self.hi.immovable = false;
                }
            }
        }
    }

    pub fn sin_assign(&mut self, x: &Ival) {
        self.err = x.err;
        let (xlo, xhi) = (x.lo.as_float(), x.hi.as_float());

        match classify_ival_periodic(x, 1) {
            PeriodClass::TooWide => {
                self.lo.as_float_mut().assign(-1);
                self.hi.as_float_mut().assign(1);
                self.lo.immovable = false;
                self.hi.immovable = false;
            }
            PeriodClass::NearZero => {
                self.monotonic_assign(&mpfr_sin, x);
            }
            PeriodClass::RangeReduce => {
                let prec = range_reduce_precision(xlo, xhi, x.prec());
                let (a, b) = ival_div_pi(x, prec, mpfr_round_inplace);

                if a == b && bfeven(&a) {
                    self.monotonic_assign(&mpfr_sin, x);
                } else if a == b && bfodd(&a) {
                    self.comonotonic_assign(&mpfr_sin, x);
                } else if bfsub_is_one(&a, &b) && bfodd(&a) {
                    self.set_prec(x.prec());
                    self.lo.as_float_mut().assign(-1);
                    self.lo.immovable = false;
                    self.hi.immovable =
                        endpoint_max(&mpfr_sin, &x.lo, &x.hi, self.hi.as_float_mut());
                } else if bfsub_is_one(&a, &b) && bfeven(&a) {
                    self.set_prec(x.prec());
                    self.lo.immovable =
                        endpoint_min(&mpfr_sin, &x.lo, &x.hi, self.lo.as_float_mut());
                    self.hi.as_float_mut().assign(1);
                    self.hi.immovable = false;
                } else {
                    self.lo.as_float_mut().assign(-1);
                    self.hi.as_float_mut().assign(1);
                    self.lo.immovable = false;
                    self.hi.immovable = false;
                }
            }
        }
    }

    pub fn tan_assign(&mut self, x: &Ival) {
        let (xlo, xhi) = (x.lo.as_float(), x.hi.as_float());
        let immovable = x.lo.immovable && x.hi.immovable;

        match classify_ival_periodic(x, 0) {
            PeriodClass::TooWide => {
                self.lo.as_float_mut().assign(f64::NEG_INFINITY);
                self.hi.as_float_mut().assign(f64::INFINITY);
                self.lo.immovable = immovable;
                self.hi.immovable = immovable;
                self.err.partial = true;
                self.err.total = x.err.total;
            }
            PeriodClass::NearZero => {
                self.monotonic_assign(&mpfr_tan, x);
                self.err = x.err;
            }
            PeriodClass::RangeReduce => {
                let prec = range_reduce_precision(xlo, xhi, x.prec());
                let (a, b) = ival_div_pi(x, prec, mpfr_round_inplace);

                if a == b {
                    self.monotonic_assign(&mpfr_tan, x);
                    self.err = x.err;
                } else {
                    self.lo.as_float_mut().assign(f64::NEG_INFINITY);
                    self.hi.as_float_mut().assign(f64::INFINITY);
                    self.lo.immovable = immovable;
                    self.hi.immovable = immovable;
                    self.err.partial = true;
                    self.err.total = x.err.total;
                }
            }
        }
    }

    pub fn cosu_assign(&mut self, x: &Ival, n: u64) {
        self.err = x.err;
        let (xlo, xhi) = (x.lo.as_float(), x.hi.as_float());
        let period_qtr = period_quarter_bitlen(n, 4);
        let cosu = |x: &Float, out: &mut Float, rnd: Round| mpfr_cosu(x, n, out, rnd);

        match classify_ival_periodic(x, period_qtr) {
            PeriodClass::TooWide => {
                self.lo.as_float_mut().assign(-1);
                self.hi.as_float_mut().assign(1);
                self.lo.immovable = false;
                self.hi.immovable = false;
            }
            PeriodClass::NearZero => match classify(x, false) {
                IvalClass::Neg => self.monotonic_assign(&cosu, x),
                IvalClass::Pos => self.comonotonic_assign(&cosu, x),
                IvalClass::Mix => {
                    self.set_prec(x.prec());
                    self.lo.immovable = endpoint_min(&cosu, &x.lo, &x.hi, self.lo.as_float_mut());
                    self.hi.as_float_mut().assign(1);
                    self.hi.immovable = false;
                }
            },
            PeriodClass::RangeReduce => {
                let prec = range_reduce_precision(xlo, xhi, x.prec());
                let (a, b) = ival_div_n_half(x, n, prec, mpfr_floor_inplace);

                if a == b && bfeven(&a) {
                    self.comonotonic_assign(&cosu, x);
                } else if a == b && bfodd(&a) {
                    self.monotonic_assign(&cosu, x);
                } else if bfsub_is_one(&a, &b) && bfeven(&a) {
                    self.set_prec(x.prec());
                    self.lo.as_float_mut().assign(-1);
                    self.lo.immovable = false;
                    self.hi.immovable = endpoint_max(&cosu, &x.lo, &x.hi, self.hi.as_float_mut());
                } else if bfsub_is_one(&a, &b) && bfodd(&a) {
                    self.set_prec(x.prec());
                    self.lo.immovable = endpoint_min(&cosu, &x.lo, &x.hi, self.lo.as_float_mut());
                    self.hi.as_float_mut().assign(1);
                    self.hi.immovable = false;
                } else {
                    self.lo.as_float_mut().assign(-1);
                    self.hi.as_float_mut().assign(1);
                    self.lo.immovable = false;
                    self.hi.immovable = false;
                }
            }
        }
    }

    pub fn sinu_assign(&mut self, x: &Ival, n: u64) {
        self.err = x.err;
        let (xlo, xhi) = (x.lo.as_float(), x.hi.as_float());
        let period_qtr = period_quarter_bitlen(n, 4);
        let sinu = |x: &Float, out: &mut Float, rnd: Round| mpfr_sinu(x, n, out, rnd);

        match classify_ival_periodic(x, period_qtr) {
            PeriodClass::TooWide => {
                self.lo.as_float_mut().assign(-1);
                self.hi.as_float_mut().assign(1);
                self.lo.immovable = false;
                self.hi.immovable = false;
            }
            PeriodClass::NearZero => {
                self.monotonic_assign(&sinu, x);
            }
            PeriodClass::RangeReduce => {
                let prec = range_reduce_precision(xlo, xhi, x.prec());
                let (a, b) = ival_div_n_half(x, n, prec, mpfr_round_inplace);

                if a == b && bfeven(&a) {
                    self.monotonic_assign(&sinu, x);
                } else if a == b && bfodd(&a) {
                    self.comonotonic_assign(&sinu, x);
                } else if bfsub_is_one(&a, &b) && bfodd(&a) {
                    self.set_prec(x.prec());
                    self.lo.as_float_mut().assign(-1);
                    self.lo.immovable = false;
                    self.hi.immovable = endpoint_max(&sinu, &x.lo, &x.hi, self.hi.as_float_mut());
                } else if bfsub_is_one(&a, &b) && bfeven(&a) {
                    self.set_prec(x.prec());
                    self.lo.immovable = endpoint_min(&sinu, &x.lo, &x.hi, self.lo.as_float_mut());
                    self.hi.as_float_mut().assign(1);
                    self.hi.immovable = false;
                } else {
                    self.lo.as_float_mut().assign(-1);
                    self.hi.as_float_mut().assign(1);
                    self.lo.immovable = false;
                    self.hi.immovable = false;
                }
            }
        }
    }

    pub fn tanu_assign(&mut self, x: &Ival, n: u64) {
        let (xlo, xhi) = (x.lo.as_float(), x.hi.as_float());
        let period_qtr = period_quarter_bitlen(n, 8);
        let immovable = x.lo.immovable && x.hi.immovable;
        let tanu = |x: &Float, out: &mut Float, rnd: Round| mpfr_tanu(x, n, out, rnd);

        match classify_ival_periodic(x, period_qtr) {
            PeriodClass::TooWide => {
                self.lo.as_float_mut().assign(f64::NEG_INFINITY);
                self.hi.as_float_mut().assign(f64::INFINITY);
                self.lo.immovable = immovable;
                self.hi.immovable = immovable;
                self.err.partial = true;
                self.err.total = x.err.total;
            }
            PeriodClass::NearZero => {
                self.monotonic_assign(&tanu, x);
                self.err = x.err;
            }
            PeriodClass::RangeReduce => {
                let prec = range_reduce_precision(xlo, xhi, x.prec());
                let (a, b) = ival_div_n_half(x, n, prec, mpfr_round_inplace);

                if a == b {
                    self.monotonic_assign(&tanu, x);
                    self.err = x.err;
                } else {
                    self.lo.as_float_mut().assign(f64::NEG_INFINITY);
                    self.hi.as_float_mut().assign(f64::INFINITY);
                    self.lo.immovable = immovable;
                    self.hi.immovable = immovable;
                    self.err.partial = true;
                    self.err.total = x.err.total;
                }
            }
        }
    }
}
