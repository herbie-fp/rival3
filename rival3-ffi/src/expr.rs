use crate::{
    BinaryOpCode, RivalError, TernaryOpCode, UnaryOpCode, UnaryParamOpCode, c_slice, status,
};
use num_enum::TryFromPrimitive;
use rival::{Expression, ExpressionBuilder};
use rug::{Float, Integer, Rational};
use std::collections::HashSet;
use std::ffi::{CStr, c_char};

pub type RivalExpr = u32;

pub const RIVAL_EXPR_INVALID: RivalExpr = u32::MAX;

pub struct RivalExprBuilder {
    pub(crate) expressions: ExpressionBuilder,
}

impl RivalExprBuilder {
    fn expression(&self, handle: RivalExpr) -> Option<Expression> {
        self.expressions.expression(handle as usize)
    }

    pub(crate) fn outputs(&self, handles: &[RivalExpr]) -> Option<Vec<Expression>> {
        handles
            .iter()
            .map(|&handle| self.expression(handle))
            .collect()
    }
}

fn build(
    builder: Option<&mut RivalExprBuilder>,
    make: impl FnOnce(&mut RivalExprBuilder) -> Option<Expression>,
) -> RivalExpr {
    builder
        .and_then(make)
        .and_then(|expression| RivalExpr::try_from(expression.index()).ok())
        .unwrap_or(RIVAL_EXPR_INVALID)
}

unsafe fn c_str<'a>(s: *const c_char) -> Option<&'a str> {
    if s.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(s) }.to_str().ok()
}

unsafe fn c_integer(s: *const c_char) -> Option<Integer> {
    Integer::parse(unsafe { c_str(s) }?).ok().map(Integer::from)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_expr_builder_new(
    vars: *const *const c_char,
    n_vars: usize,
    out: *mut *mut RivalExprBuilder,
) -> RivalError {
    status(|| {
        let out = unsafe { out.as_mut() }.ok_or(RivalError::Misuse)?;
        let vars = unsafe { c_slice(vars, n_vars)? };
        let arguments = vars
            .iter()
            .map(|&var| unsafe { c_str(var) })
            .collect::<Option<Vec<_>>>()
            .ok_or(RivalError::Misuse)?;

        // Must have unique variable names.
        let mut seen = HashSet::new();
        if !arguments.iter().all(|name| seen.insert(*name)) {
            return Err(RivalError::Misuse);
        }

        let expressions = ExpressionBuilder::new(arguments);
        *out = Box::into_raw(Box::new(RivalExprBuilder { expressions }));
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_expr_builder_free(builder: *mut RivalExprBuilder) {
    if !builder.is_null() {
        unsafe { drop(Box::from_raw(builder)) };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_expr_var(
    builder: *mut RivalExprBuilder,
    name: *const c_char,
) -> RivalExpr {
    let name = unsafe { c_str(name) };
    build(unsafe { builder.as_mut() }, |b| {
        b.expressions.variable(name?)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_expr_f64(builder: *mut RivalExprBuilder, value: f64) -> RivalExpr {
    build(unsafe { builder.as_mut() }, |b| {
        Some(
            b.expressions
                .literal(Float::with_val(f64::MANTISSA_DIGITS, value)),
        )
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_expr_bigint(
    builder: *mut RivalExprBuilder,
    value: *const c_char,
) -> RivalExpr {
    let value = unsafe { c_integer(value) };
    build(unsafe { builder.as_mut() }, |b| {
        Some(b.expressions.rational(Rational::from(value?)))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_expr_bigrational(
    builder: *mut RivalExprBuilder,
    numerator: *const c_char,
    denominator: *const c_char,
) -> RivalExpr {
    let numerator = unsafe { c_integer(numerator) };
    let denominator = unsafe { c_integer(denominator) };
    build(unsafe { builder.as_mut() }, |b| {
        let (numerator, denominator) = (numerator?, denominator?);
        (denominator != 0).then(|| {
            b.expressions
                .rational(Rational::from((numerator, denominator)))
        })
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_expr_pi(builder: *mut RivalExprBuilder) -> RivalExpr {
    build(unsafe { builder.as_mut() }, |b| Some(b.expressions.pi()))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_expr_e(builder: *mut RivalExprBuilder) -> RivalExpr {
    build(unsafe { builder.as_mut() }, |b| Some(b.expressions.e()))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_expr_infinity(builder: *mut RivalExprBuilder) -> RivalExpr {
    build(unsafe { builder.as_mut() }, |b| {
        Some(b.expressions.infinity())
    })
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, TryFromPrimitive)]
pub enum RivalUnaryOp {
    Neg = 0,
    Fabs = 1,
    Sqrt = 2,
    Cbrt = 3,
    Pow2 = 4,
    Exp = 5,
    Exp2 = 6,
    Expm1 = 7,
    Log = 8,
    Log2 = 9,
    Log10 = 10,
    Log1p = 11,
    Logb = 12,
    Sin = 13,
    Cos = 14,
    Tan = 15,
    Asin = 16,
    Acos = 17,
    Atan = 18,
    Sinh = 19,
    Cosh = 20,
    Tanh = 21,
    Asinh = 22,
    Acosh = 23,
    Atanh = 24,
    Erf = 25,
    Erfc = 26,
    Lgamma = 27,
    Tgamma = 28,
    Rint = 29,
    Round = 30,
    Ceil = 31,
    Floor = 32,
    Trunc = 33,
    Not = 34,
    Assert = 35,
    Error = 36,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, TryFromPrimitive)]
pub enum RivalUnaryParamOp {
    Cosu = 0,
    Sinu = 1,
    Tanu = 2,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, TryFromPrimitive)]
pub enum RivalBinaryOp {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    Pow = 4,
    Hypot = 5,
    Fmin = 6,
    Fmax = 7,
    Fdim = 8,
    Copysign = 9,
    Fmod = 10,
    Remainder = 11,
    Atan2 = 12,
    And = 13,
    Or = 14,
    Eq = 15,
    Ne = 16,
    Lt = 17,
    Le = 18,
    Gt = 19,
    Ge = 20,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, TryFromPrimitive)]
pub enum RivalTernaryOp {
    Fma = 0,
    If = 1,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_expr_unary(
    builder: *mut RivalExprBuilder,
    op: UnaryOpCode,
    arg: RivalExpr,
) -> RivalExpr {
    build(unsafe { builder.as_mut() }, |b| {
        let arg = b.expression(arg)?;
        let expressions = &mut b.expressions;
        Some(match RivalUnaryOp::try_from(op).ok()? {
            RivalUnaryOp::Neg => expressions.neg(arg),
            RivalUnaryOp::Fabs => expressions.fabs(arg),
            RivalUnaryOp::Sqrt => expressions.sqrt(arg),
            RivalUnaryOp::Cbrt => expressions.cbrt(arg),
            RivalUnaryOp::Pow2 => expressions.pow2(arg),
            RivalUnaryOp::Exp => expressions.exp(arg),
            RivalUnaryOp::Exp2 => expressions.exp2(arg),
            RivalUnaryOp::Expm1 => expressions.expm1(arg),
            RivalUnaryOp::Log => expressions.log(arg),
            RivalUnaryOp::Log2 => expressions.log2(arg),
            RivalUnaryOp::Log10 => expressions.log10(arg),
            RivalUnaryOp::Log1p => expressions.log1p(arg),
            RivalUnaryOp::Logb => expressions.logb(arg),
            RivalUnaryOp::Sin => expressions.sin(arg),
            RivalUnaryOp::Cos => expressions.cos(arg),
            RivalUnaryOp::Tan => expressions.tan(arg),
            RivalUnaryOp::Asin => expressions.asin(arg),
            RivalUnaryOp::Acos => expressions.acos(arg),
            RivalUnaryOp::Atan => expressions.atan(arg),
            RivalUnaryOp::Sinh => expressions.sinh(arg),
            RivalUnaryOp::Cosh => expressions.cosh(arg),
            RivalUnaryOp::Tanh => expressions.tanh(arg),
            RivalUnaryOp::Asinh => expressions.asinh(arg),
            RivalUnaryOp::Acosh => expressions.acosh(arg),
            RivalUnaryOp::Atanh => expressions.atanh(arg),
            RivalUnaryOp::Erf => expressions.erf(arg),
            RivalUnaryOp::Erfc => expressions.erfc(arg),
            RivalUnaryOp::Lgamma => expressions.lgamma(arg),
            RivalUnaryOp::Tgamma => expressions.tgamma(arg),
            RivalUnaryOp::Rint => expressions.rint(arg),
            RivalUnaryOp::Round => expressions.round(arg),
            RivalUnaryOp::Ceil => expressions.ceil(arg),
            RivalUnaryOp::Floor => expressions.floor(arg),
            RivalUnaryOp::Trunc => expressions.trunc(arg),
            RivalUnaryOp::Not => expressions.not(arg),
            RivalUnaryOp::Assert => expressions.assert(arg),
            RivalUnaryOp::Error => expressions.error(arg),
        })
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_expr_unary_param(
    builder: *mut RivalExprBuilder,
    op: UnaryParamOpCode,
    param: u32,
    arg: RivalExpr,
) -> RivalExpr {
    build(unsafe { builder.as_mut() }, |b| {
        let arg = b.expression(arg)?;
        let expressions = &mut b.expressions;
        let param = param.into();
        Some(match RivalUnaryParamOp::try_from(op).ok()? {
            RivalUnaryParamOp::Cosu => expressions.cosu(param, arg),
            RivalUnaryParamOp::Sinu => expressions.sinu(param, arg),
            RivalUnaryParamOp::Tanu => expressions.tanu(param, arg),
        })
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_expr_binary(
    builder: *mut RivalExprBuilder,
    op: BinaryOpCode,
    lhs: RivalExpr,
    rhs: RivalExpr,
) -> RivalExpr {
    build(unsafe { builder.as_mut() }, |b| {
        let (lhs, rhs) = (b.expression(lhs)?, b.expression(rhs)?);
        let expressions = &mut b.expressions;
        Some(match RivalBinaryOp::try_from(op).ok()? {
            RivalBinaryOp::Add => expressions.add(lhs, rhs),
            RivalBinaryOp::Sub => expressions.sub(lhs, rhs),
            RivalBinaryOp::Mul => expressions.mul(lhs, rhs),
            RivalBinaryOp::Div => expressions.div(lhs, rhs),
            RivalBinaryOp::Pow => expressions.pow(lhs, rhs),
            RivalBinaryOp::Hypot => expressions.hypot(lhs, rhs),
            RivalBinaryOp::Fmin => expressions.fmin(lhs, rhs),
            RivalBinaryOp::Fmax => expressions.fmax(lhs, rhs),
            RivalBinaryOp::Fdim => expressions.fdim(lhs, rhs),
            RivalBinaryOp::Copysign => expressions.copysign(lhs, rhs),
            RivalBinaryOp::Fmod => expressions.fmod(lhs, rhs),
            RivalBinaryOp::Remainder => expressions.remainder(lhs, rhs),
            RivalBinaryOp::Atan2 => expressions.atan2(lhs, rhs),
            RivalBinaryOp::And => expressions.and(lhs, rhs),
            RivalBinaryOp::Or => expressions.or(lhs, rhs),
            RivalBinaryOp::Eq => expressions.eq(lhs, rhs),
            RivalBinaryOp::Ne => expressions.ne(lhs, rhs),
            RivalBinaryOp::Lt => expressions.lt(lhs, rhs),
            RivalBinaryOp::Le => expressions.le(lhs, rhs),
            RivalBinaryOp::Gt => expressions.gt(lhs, rhs),
            RivalBinaryOp::Ge => expressions.ge(lhs, rhs),
        })
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rival_expr_ternary(
    builder: *mut RivalExprBuilder,
    op: TernaryOpCode,
    arg1: RivalExpr,
    arg2: RivalExpr,
    arg3: RivalExpr,
) -> RivalExpr {
    build(unsafe { builder.as_mut() }, |b| {
        let (arg1, arg2, arg3) = (
            b.expression(arg1)?,
            b.expression(arg2)?,
            b.expression(arg3)?,
        );
        let expressions = &mut b.expressions;
        Some(match RivalTernaryOp::try_from(op).ok()? {
            RivalTernaryOp::Fma => expressions.fma(arg1, arg2, arg3),
            RivalTernaryOp::If => expressions.if_else(arg1, arg2, arg3),
        })
    })
}
