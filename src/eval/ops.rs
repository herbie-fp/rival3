//! Operation registry generating evaluation and optimization helpers
//! Defines interval operators along with dispatch, bounds, and path reduction hooks
use crate::eval::adjust::path_reduction;
use crate::eval::macros::def_ops;
use crate::eval::tricks::{TrickContext, crosses_zero, get_slack};
use crate::interval::Ival;

def_ops! {
    constant {
        Pi: {
            method: set_pi,
        },

        E: {
            method: set_e,
        },

        Infinity: {
            method: set_infinity,
        },
    },

    unary {
        Pow2: {
            method: pow2_assign,
            bounds: |ctx, _out, inp| {
                ctx.logspan(inp) + 1
            },
        },
        Fabs: {
            method: fabs_assign,
            bounds: |_, _, _| 0,
        },

        Neg: {
            method: neg_assign,
            bounds: |_, _, _| 0,
        },

        Sqrt: {
            method: sqrt_assign,
            bounds: |ctx, _, inp| (ctx.logspan(inp) / 2).saturating_sub(1),
        },

        Cbrt: {
            method: cbrt_assign,
            bounds: |ctx, _, inp| ((2 * ctx.logspan(inp)) / 3).saturating_sub(1),
        },

        Exp: {
            method: exp_assign,
            bounds: |ctx, out, inp| {
                ctx.maxlog(inp) + ctx.logspan(out)
            },
        },

        Exp2: {
            method: exp2_assign,
            bounds: |ctx, out, inp| {
                ctx.maxlog(inp) + ctx.logspan(out)
            },
        },

        Expm1: {
            method: expm1_assign,
            bounds: |ctx, out, inp| {
                let mx = ctx.maxlog(inp);
                (1 + mx).max(1 + mx - ctx.minlog(out))
            },
        },

        Log: {
            method: log_assign,
            bounds: |ctx, out, inp| {
                ctx.logspan(inp) - ctx.minlog(out) + 1
            },
        },

        Log2: {
            method: log2_assign,
            bounds: |ctx, out, inp| {
                ctx.logspan(inp) - ctx.minlog(out) + 1
            },
        },

        Log10: {
            method: log10_assign,
            bounds: |ctx, out, inp| {
                ctx.logspan(inp) - ctx.minlog(out) + 1
            },
        },

        Log1p: {
            method: log1p_assign,
            bounds: |ctx, out, inp| {
                let upper_base = ctx.maxlog(inp) - ctx.minlog(out);
                let lo_neg = inp.lo.as_float().is_sign_negative();
                let hi_neg = inp.hi.as_float().is_sign_negative();
                let slack = if lo_neg || hi_neg { get_slack(ctx.iteration, ctx.slack_unit) } else { 0 };
                upper_base + slack
            },
        },

        Logb: {
            method: logb_assign,
            bounds: |ctx, _, _| get_slack(ctx.iteration, ctx.slack_unit),
        },

        Sin: {
            method: sin_assign,
            bounds: |ctx, out, inp| {
                ctx.maxlog(inp) - ctx.minlog(out)
            },
        },

        Cos: {
            method: cos_assign,
            bounds: |ctx, out, inp| {
                ctx.maxlog(inp) - ctx.minlog(out)
                    + ctx.maxlog(inp).min(0)
            },
        },

        Tan: {
            method: tan_assign,
            bounds: |ctx, out, inp| {
                ctx.maxlog(inp)
                    + ctx.maxlog(out).abs().max(ctx.minlog(out).abs())
                    + ctx.logspan(out)
                    + 1
            },
        },

        Asin: {
            method: asin_assign,
            bounds: |ctx, out, _| {
                if ctx.maxlog(out) >= 1 { get_slack(ctx.iteration, ctx.slack_unit) } else { 1 }
            },
        },

        Acos: {
            method: acos_assign,
            bounds: |ctx, out, inp| {
                let maxlog_inp = ctx.maxlog(inp);
                let slack = if maxlog_inp >= 0 { get_slack(ctx.iteration, ctx.slack_unit) } else { 0 };
                maxlog_inp - ctx.minlog(out) + slack
            },
        },

        Atan: {
            method: atan_assign,
            bounds: |ctx, out, inp| {
                ctx.logspan(inp)
                    - ctx.minlog(inp).abs().min(ctx.maxlog(inp).abs())
                    - ctx.minlog(out)
            },
        },

        Sinh: {
            method: sinh_assign,
            bounds: |ctx, out, inp| {
                ctx.maxlog(inp) + ctx.logspan(out) - ctx.minlog(inp).min(0)
            },
        },

        Cosh: {
            method: cosh_assign,
            bounds: |ctx, out, inp| {
                ctx.maxlog(inp) + ctx.logspan(out) + ctx.maxlog(inp).min(0)
            },
        },

        Tanh: {
            method: tanh_assign,
            bounds: |ctx, out, inp| {
                ctx.logspan(out) + ctx.logspan(inp)
            },
        },

        Asinh: {
            method: asinh_assign,
            bounds: |ctx, _, _| get_slack(ctx.iteration, ctx.slack_unit),
        },

        Acosh: {
            method: acosh_assign,
            bounds: |ctx, out, _| {
                let z_exp = ctx.minlog(out);
                if z_exp < 2 { get_slack(ctx.iteration, ctx.slack_unit) - z_exp } else { 0 }
            },
        },

        Atanh: {
            method: atanh_assign,
            bounds: |ctx, _out, inp| {
                if ctx.maxlog(inp) >= 1 { get_slack(ctx.iteration, ctx.slack_unit) } else { 1 }
            },
        },

        Erf: {
            method: erf_assign,
            bounds: |ctx, _, _| get_slack(ctx.iteration, ctx.slack_unit),
        },

        Erfc: {
            method: erfc_assign,
            bounds: |ctx, _, _| get_slack(ctx.iteration, ctx.slack_unit),
        },

        Lgamma: {
            method: lgamma_assign,
            bounds: |ctx, _, _| get_slack(ctx.iteration, ctx.slack_unit),
        },

        Tgamma: {
            method: tgamma_assign,
            bounds: |ctx, _, _| get_slack(ctx.iteration, ctx.slack_unit),
        },

        Rint: {
            method: rint_assign,
            bounds: |ctx, _, _| get_slack(ctx.iteration, ctx.slack_unit),
        },

        Round: {
            method: round_assign,
            bounds: |ctx, _, _| get_slack(ctx.iteration, ctx.slack_unit),
        },

        Ceil: {
            method: ceil_assign,
            bounds: |ctx, _, _| get_slack(ctx.iteration, ctx.slack_unit),
        },

        Floor: {
            method: floor_assign,
            bounds: |ctx, _, _| get_slack(ctx.iteration, ctx.slack_unit),
        },

        Trunc: {
            method: trunc_assign,
            bounds: |ctx, _, _| get_slack(ctx.iteration, ctx.slack_unit),
        },

        Not: {
            method: not_assign,
            bounds: |_, _, _| 0,
            path_reduce: path_reduction::not_op_path_reduce,
        },

        Error: {
            method: error_assign,
            bounds: |_, _, _| 0,
        },

        Assert: {
            method: assert_assign,
            bounds: |_, _, _| 0,
            path_reduce: path_reduction::assert_op_path_reduce,
        },
    },

    unary_param {
        Cosu: {
            method: cosu_assign,
            bounds: |ctx, param, out, inp| {
                let n_log = param as i64;
                ctx.maxlog(inp) - n_log - ctx.minlog(out) + 2
            },
        },

        Sinu: {
            method: sinu_assign,
            bounds: |ctx, param, out, inp| {
                let n_log = param as i64;
                ctx.maxlog(inp) - n_log - ctx.minlog(out) + 2
            },
        },

        Tanu: {
            method: tanu_assign,
            bounds: |ctx, param, out, inp| {
                let n_log = param as i64;
                ctx.maxlog(inp) - n_log
                    + ctx.maxlog(out).abs().max(ctx.minlog(out).abs()) + 3
            },
        },
    },

    binary {
        Pow: {
            method: pow_assign,
            bounds: |ctx, out, x, y| {
                let maxlog_y = ctx.maxlog(y);
                let logspan_x = ctx.logspan(x);
                let logspan_out = ctx.logspan(out);
                let maxlog_x = ctx.maxlog(x);
                let minlog_x = ctx.minlog(x);
                // Slack adjustments
                let y_slack = if crosses_zero(out) && x.lo.as_float().is_sign_negative() { get_slack(ctx.iteration, ctx.slack_unit) } else { 0 };
                let x_slack = if out.lo.as_float().is_zero() { get_slack(ctx.iteration, ctx.slack_unit) } else { 0 };

                // Upper bounds
                let upper_x_base = maxlog_y + logspan_x + logspan_out + x_slack;
                let upper_x = upper_x_base.max(x_slack);
                let abs_maxlog_x = maxlog_x.abs();
                let abs_minlog_x = minlog_x.abs();
                let span_x_mag = abs_maxlog_x.max(abs_minlog_x);
                let upper_y_base = maxlog_y + span_x_mag + logspan_out + y_slack;
                let upper_y = upper_y_base.max(y_slack);

                (upper_x, upper_y)
            },
        },

        Fdim: {
            method: fdim_assign,
            bounds: |ctx, out, x, y| {
                let output_min = ctx.minlog(out);
                let lhs_upper = ctx.maxlog(x) - output_min;
                let rhs_upper = ctx.maxlog(y) - output_min;
                (lhs_upper, rhs_upper)
            },
        },

        Hypot: {
            method: hypot_assign,
            bounds: |ctx, _, _, _| {
                let bounds = get_slack(ctx.iteration, ctx.slack_unit);
                (bounds, bounds)
            },
        },
        Add: {
            method: add_assign,
            bounds: |ctx, out, lhs, rhs| {
                let output_min = ctx.minlog(out);
                let lhs_upper = ctx.maxlog(lhs) - output_min;
                let rhs_upper = ctx.maxlog(rhs) - output_min;
                (lhs_upper, rhs_upper)
            },
        },

        Sub: {
            method: sub_assign,
            bounds: |ctx, out, lhs, rhs| {
                let output_min = ctx.minlog(out);
                let lhs_upper = ctx.maxlog(lhs) - output_min;
                let rhs_upper = ctx.maxlog(rhs) - output_min;
                (lhs_upper, rhs_upper)
            },
        },

        Mul: {
            method: mul_assign,
            bounds: |ctx, _, lhs, rhs| {
                (ctx.logspan(rhs), ctx.logspan(lhs))
            },
        },

        Div: {
            method: div_assign,
            bounds: |ctx, _, lhs, rhs| {
                let lhs_bounds = ctx.logspan(rhs);
                let rhs_bounds = ctx.logspan(lhs) + 2 * ctx.logspan(rhs);
                (lhs_bounds, rhs_bounds)
            },
        },

        And: {
            method: and_assign,
            bounds: |_, _, _, _| (0, 0),
            path_reduce: path_reduction::bool_op_path_reduce,
        },

        Or: {
            method: or_assign,
            bounds: |_, _, _, _| (0, 0),
            path_reduce: path_reduction::bool_op_path_reduce,
        },

        Eq: {
            method: eq_assign,
            bounds: |_, _, _, _| (0, 0),
            path_reduce: path_reduction::bool_op_path_reduce,
        },

        Ne: {
            method: ne_assign,
            bounds: |_, _, _, _| (0, 0),
            path_reduce: path_reduction::bool_op_path_reduce,
        },

        Lt: {
            method: lt_assign,
            bounds: |_, _, _, _| (0, 0),
            path_reduce: path_reduction::bool_op_path_reduce,
        },

        Le: {
            method: le_assign,
            bounds: |_, _, _, _| (0, 0),
            path_reduce: path_reduction::bool_op_path_reduce,
        },

        Gt: {
            method: gt_assign,
            bounds: |_, _, _, _| (0, 0),
            path_reduce: path_reduction::bool_op_path_reduce,
        },

        Ge: {
            method: ge_assign,
            bounds: |_, _, _, _| (0, 0),
            path_reduce: path_reduction::bool_op_path_reduce,
        },

        Fmin: {
            method: fmin_assign,
            bounds: |_, _, _, _| (0, 0),
            path_reduce: |machine, idx, mark| {
                path_reduction::minmax_path_reduce(machine, idx, mark, false)
            },
        },

        Fmax: {
            method: fmax_assign,
            bounds: |_, _, _, _| (0, 0),
            path_reduce: |machine, idx, mark| {
                path_reduction::minmax_path_reduce(machine, idx, mark, true)
            },
        },

        Copysign: {
            method: copysign_assign,
            bounds: |_, _, _, _| (0, 0),
        },

        Atan2: {
            method: atan2_assign,
            bounds: |ctx, out, y, x| {
                let upper = ctx.maxlog(x) + ctx.maxlog(y)
                    - 2 * ctx.minlog(x).min(ctx.minlog(y))
                    - ctx.minlog(out);
                (upper, upper)
            },
        },

        Fmod: {
            method: fmod_assign,
            bounds: |ctx, out, x, y| {
                let slack = if crosses_zero(y) { get_slack(ctx.iteration, ctx.slack_unit) } else { 0 };
                let upper_x = ctx.maxlog(x) - ctx.minlog(out);
                let upper_y = upper_x + slack;
                (upper_x, upper_y)
            },
        },

        Remainder: {
            method: remainder_assign,
            bounds: |ctx, out, x, y| {
                let slack = if crosses_zero(y) { get_slack(ctx.iteration, ctx.slack_unit) } else { 0 };
                let upper_x = ctx.maxlog(x) - ctx.minlog(out);
                let upper_y = upper_x + slack;
                (upper_x, upper_y)
            },
        },
    },

    ternary {
        Fma: {
            method: fma_assign,
            bounds: |ctx, out, a, b, _c| {
                (ctx.logspan(b) + ctx.logspan(out),
                 ctx.logspan(a) + ctx.logspan(out),
                 ctx.logspan(out))
            },
        },

        If: {
            method: if_assign,
            bounds: |_, _, _, _, _| (0, 0, 0),
            path_reduce: path_reduction::if_op_path_reduce,
        },
    },
}
