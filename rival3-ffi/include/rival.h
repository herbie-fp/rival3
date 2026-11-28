#ifndef RIVAL3_FFI_H
#define RIVAL3_FFI_H

/* Generated with cbindgen:0.29.2 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <mpfr.h>

#define RIVAL_ABI_VERSION 4

enum RivalError
#ifdef __cplusplus
  : int32_t
#endif // __cplusplus
 {
    RIVAL_ERROR_OK = 0,
    RIVAL_ERROR_INVALID_INPUT = -1,
    RIVAL_ERROR_UNSAMPLABLE = -2,
    RIVAL_ERROR_MISUSE = -3,
};
#ifndef __cplusplus
typedef int32_t RivalError;
#endif // __cplusplus

enum RivalAnalysis
#ifdef __cplusplus
  : uint32_t
#endif // __cplusplus
 {
    RIVAL_ANALYSIS_VALID = 0,
    RIVAL_ANALYSIS_MAYBE_INVALID = 1,
    RIVAL_ANALYSIS_INVALID = 2,
};
#ifndef __cplusplus
typedef uint32_t RivalAnalysis;
#endif // __cplusplus

enum RivalDiscType
#ifdef __cplusplus
  : uint32_t
#endif // __cplusplus
 {
    RIVAL_DISC_TYPE_BOOL = 0,
    RIVAL_DISC_TYPE_F32 = 1,
    RIVAL_DISC_TYPE_F64 = 2,
};
#ifndef __cplusplus
typedef uint32_t RivalDiscType;
#endif // __cplusplus

enum RivalUnaryOp
#ifdef __cplusplus
  : uint32_t
#endif // __cplusplus
 {
    RIVAL_UNARY_OP_NEG = 0,
    RIVAL_UNARY_OP_FABS = 1,
    RIVAL_UNARY_OP_SQRT = 2,
    RIVAL_UNARY_OP_CBRT = 3,
    RIVAL_UNARY_OP_POW2 = 4,
    RIVAL_UNARY_OP_EXP = 5,
    RIVAL_UNARY_OP_EXP2 = 6,
    RIVAL_UNARY_OP_EXPM1 = 7,
    RIVAL_UNARY_OP_LOG = 8,
    RIVAL_UNARY_OP_LOG2 = 9,
    RIVAL_UNARY_OP_LOG10 = 10,
    RIVAL_UNARY_OP_LOG1P = 11,
    RIVAL_UNARY_OP_LOGB = 12,
    RIVAL_UNARY_OP_SIN = 13,
    RIVAL_UNARY_OP_COS = 14,
    RIVAL_UNARY_OP_TAN = 15,
    RIVAL_UNARY_OP_ASIN = 16,
    RIVAL_UNARY_OP_ACOS = 17,
    RIVAL_UNARY_OP_ATAN = 18,
    RIVAL_UNARY_OP_SINH = 19,
    RIVAL_UNARY_OP_COSH = 20,
    RIVAL_UNARY_OP_TANH = 21,
    RIVAL_UNARY_OP_ASINH = 22,
    RIVAL_UNARY_OP_ACOSH = 23,
    RIVAL_UNARY_OP_ATANH = 24,
    RIVAL_UNARY_OP_ERF = 25,
    RIVAL_UNARY_OP_ERFC = 26,
    RIVAL_UNARY_OP_LGAMMA = 27,
    RIVAL_UNARY_OP_TGAMMA = 28,
    RIVAL_UNARY_OP_RINT = 29,
    RIVAL_UNARY_OP_ROUND = 30,
    RIVAL_UNARY_OP_CEIL = 31,
    RIVAL_UNARY_OP_FLOOR = 32,
    RIVAL_UNARY_OP_TRUNC = 33,
    RIVAL_UNARY_OP_NOT = 34,
    RIVAL_UNARY_OP_ASSERT = 35,
    RIVAL_UNARY_OP_ERROR = 36,
};
#ifndef __cplusplus
typedef uint32_t RivalUnaryOp;
#endif // __cplusplus

enum RivalUnaryParamOp
#ifdef __cplusplus
  : uint32_t
#endif // __cplusplus
 {
    RIVAL_UNARY_PARAM_OP_COSU = 0,
    RIVAL_UNARY_PARAM_OP_SINU = 1,
    RIVAL_UNARY_PARAM_OP_TANU = 2,
};
#ifndef __cplusplus
typedef uint32_t RivalUnaryParamOp;
#endif // __cplusplus

enum RivalBinaryOp
#ifdef __cplusplus
  : uint32_t
#endif // __cplusplus
 {
    RIVAL_BINARY_OP_ADD = 0,
    RIVAL_BINARY_OP_SUB = 1,
    RIVAL_BINARY_OP_MUL = 2,
    RIVAL_BINARY_OP_DIV = 3,
    RIVAL_BINARY_OP_POW = 4,
    RIVAL_BINARY_OP_HYPOT = 5,
    RIVAL_BINARY_OP_FMIN = 6,
    RIVAL_BINARY_OP_FMAX = 7,
    RIVAL_BINARY_OP_FDIM = 8,
    RIVAL_BINARY_OP_COPYSIGN = 9,
    RIVAL_BINARY_OP_FMOD = 10,
    RIVAL_BINARY_OP_REMAINDER = 11,
    RIVAL_BINARY_OP_ATAN2 = 12,
    RIVAL_BINARY_OP_AND = 13,
    RIVAL_BINARY_OP_OR = 14,
    RIVAL_BINARY_OP_EQ = 15,
    RIVAL_BINARY_OP_NE = 16,
    RIVAL_BINARY_OP_LT = 17,
    RIVAL_BINARY_OP_LE = 18,
    RIVAL_BINARY_OP_GT = 19,
    RIVAL_BINARY_OP_GE = 20,
};
#ifndef __cplusplus
typedef uint32_t RivalBinaryOp;
#endif // __cplusplus

enum RivalTernaryOp
#ifdef __cplusplus
  : uint32_t
#endif // __cplusplus
 {
    RIVAL_TERNARY_OP_FMA = 0,
    RIVAL_TERNARY_OP_IF = 1,
};
#ifndef __cplusplus
typedef uint32_t RivalTernaryOp;
#endif // __cplusplus

enum RivalStrategy
#ifdef __cplusplus
  : uint32_t
#endif // __cplusplus
 {
    RIVAL_STRATEGY_ADAPTIVE = 0,
    RIVAL_STRATEGY_BASELINE = 1,
};
#ifndef __cplusplus
typedef uint32_t RivalStrategy;
#endif // __cplusplus

enum RivalOutputPolicy
#ifdef __cplusplus
  : uint32_t
#endif // __cplusplus
 {
    RIVAL_OUTPUT_POLICY_REQUIRE_ALL = 0,
    RIVAL_OUTPUT_POLICY_ALLOW_PARTIAL = 1,
};
#ifndef __cplusplus
typedef uint32_t RivalOutputPolicy;
#endif // __cplusplus

enum RivalInputKind
#ifdef __cplusplus
  : uint32_t
#endif // __cplusplus
 {
    RIVAL_INPUT_KIND_TEXT = 0,
    RIVAL_INPUT_KIND_F64 = 1,
    RIVAL_INPUT_KIND_BOOL = 2,
    RIVAL_INPUT_KIND_I64 = 3,
    RIVAL_INPUT_KIND_U64 = 4,
    RIVAL_INPUT_KIND_MPFR = 5,
    RIVAL_INPUT_KIND_RATIONAL = 6,
};
#ifndef __cplusplus
typedef uint32_t RivalInputKind;
#endif // __cplusplus

typedef struct RivalExprBuilder RivalExprBuilder;

typedef struct RivalHints RivalHints;

typedef struct RivalMachine RivalMachine;

typedef struct RivalMachineOptions RivalMachineOptions;

typedef uint32_t RivalExpr;

typedef struct {
    int64_t numerator;
    uint64_t denominator;
} RivalRational;

typedef union {
    const char *text;
    double f64_value;
    bool bool_value;
    int64_t i64_value;
    uint64_t u64_value;
    mpfr_srcptr mpfr_value;
    const RivalRational *rational_value;
} RivalInputValue;

typedef struct {
    uint32_t kind;
    RivalInputValue value;
} RivalInput;

typedef struct {
    int32_t instruction_idx;
    uint32_t precision;
    double time_ms;
    uint32_t iteration;
} RivalExecution;

#define RIVAL_EXPR_INVALID UINT32_MAX

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

uint32_t rival_version(void);

RivalError rival_expr_builder_new(const char *const *vars, size_t n_vars, RivalExprBuilder **out);

void rival_expr_builder_free(RivalExprBuilder *builder);

RivalExpr rival_expr_var(RivalExprBuilder *builder, const char *name);

RivalExpr rival_expr_f64(RivalExprBuilder *builder, double value);

RivalExpr rival_expr_bigint(RivalExprBuilder *builder, const char *value);

RivalExpr rival_expr_bigrational(RivalExprBuilder *builder,
                                 const char *numerator,
                                 const char *denominator);

RivalExpr rival_expr_pi(RivalExprBuilder *builder);

RivalExpr rival_expr_e(RivalExprBuilder *builder);

RivalExpr rival_expr_infinity(RivalExprBuilder *builder);

RivalExpr rival_expr_unary(RivalExprBuilder *builder, RivalUnaryOp op, RivalExpr arg);

RivalExpr rival_expr_unary_param(RivalExprBuilder *builder,
                                 RivalUnaryParamOp op,
                                 uint32_t param,
                                 RivalExpr arg);

RivalExpr rival_expr_binary(RivalExprBuilder *builder,
                            RivalBinaryOp op,
                            RivalExpr lhs,
                            RivalExpr rhs);

RivalExpr rival_expr_ternary(RivalExprBuilder *builder,
                             RivalTernaryOp op,
                             RivalExpr arg1,
                             RivalExpr arg2,
                             RivalExpr arg3);

RivalError rival_machine_new(const RivalExprBuilder *builder,
                             const RivalExpr *expr_handles,
                             const RivalDiscType *types,
                             size_t n_exprs,
                             const RivalMachineOptions *options,
                             RivalMachine **out);

void rival_machine_free(RivalMachine *machine);

void rival_hints_free(RivalHints *hints);

size_t rival_machine_instruction_count(const RivalMachine *machine);

size_t rival_machine_var_count(const RivalMachine *machine);

size_t rival_machine_expr_count(const RivalMachine *machine);

/**
 * Apply MPFR point inputs. Adaptive evaluation stops when refinement cannot progress.
 */
RivalError rival_machine_apply_mpfr_to_mpfr(RivalMachine *machine,
                                            const mpfr_srcptr *args,
                                            size_t n_args,
                                            const RivalHints *hints,
                                            RivalOutputPolicy policy,
                                            mpfr_srcptr *out,
                                            size_t n_out);

/**
 * Apply typed point values. Text inputs are parsed by Rust at each
 * instruction's working precision.
 */
RivalError rival_machine_apply_any_to_mpfr(RivalMachine *machine,
                                           const RivalInput *args,
                                           size_t n_args,
                                           const RivalHints *hints,
                                           RivalOutputPolicy policy,
                                           mpfr_srcptr *out,
                                           size_t n_out);

/**
 * Apply binary64 inputs and return full-working-precision MPFR endpoints.
 * Output pointers are borrowed from the machine, as with `rival_machine_apply_any_to_mpfr`.
 */
RivalError rival_machine_apply_f64_to_mpfr(RivalMachine *machine,
                                           const double *args,
                                           size_t n_args,
                                           const RivalHints *hints,
                                           RivalOutputPolicy policy,
                                           mpfr_srcptr *out,
                                           size_t n_out);

/**
 * Apply binary64 point inputs.
 */
RivalError rival_machine_apply_f64_to_f64(RivalMachine *machine,
                                          const double *args,
                                          size_t n_args,
                                          const RivalHints *hints,
                                          RivalOutputPolicy policy,
                                          double *out,
                                          size_t n_out);

RivalError rival_machine_analyze_mpfr(RivalMachine *machine,
                                      const mpfr_srcptr *rect,
                                      size_t n_args,
                                      const RivalHints *hints,
                                      RivalOutputPolicy policy,
                                      RivalAnalysis *out_analysis,
                                      bool *out_converged,
                                      RivalHints **out_hints);

/**
 * Analyze typed bounds supplied as `[lo0, hi0, lo1, hi1, ...]`.
 * Text values are parsed as exact decimal or integer-fraction numerals. Other
 * kinds carry native f64, bool, integer, or MPFR values.
 */
RivalError rival_machine_analyze(RivalMachine *machine,
                                 const RivalInput *bounds,
                                 size_t n_args,
                                 const RivalHints *hints,
                                 RivalOutputPolicy policy,
                                 RivalAnalysis *out_analysis,
                                 bool *out_converged,
                                 RivalHints **out_hints);

RivalError rival_machine_analyze_f64(RivalMachine *machine,
                                     const double *bounds,
                                     size_t n_args,
                                     const RivalHints *hints,
                                     RivalOutputPolicy policy,
                                     RivalAnalysis *out_analysis,
                                     bool *out_converged,
                                     RivalHints **out_hints);

/**
 * Enable the adaptive upper-bound precision-cap early exit at compilation.
 * Disabled by default; baseline evaluation ignores this option.
 */
RivalError rival_machine_options_set_early_exit(RivalMachineOptions *options, bool enabled);

RivalError rival_machine_options_new(RivalMachineOptions **out);

void rival_machine_options_free(RivalMachineOptions *options);

RivalError rival_machine_options_set_target_precision(RivalMachineOptions *options,
                                                      uint32_t precision);

RivalError rival_machine_options_set_max_precision(RivalMachineOptions *options,
                                                   uint32_t precision);

RivalError rival_machine_options_set_profile_capacity(RivalMachineOptions *options,
                                                      size_t capacity);

RivalError rival_machine_options_set_strategy(RivalMachineOptions *options, RivalStrategy strategy);

RivalError rival_machine_set_profiling(RivalMachine *machine, bool enabled);

bool rival_machine_get_profiling(const RivalMachine *machine);

RivalError rival_machine_profiler_reset(RivalMachine *machine);

RivalError rival_machine_profiler_executions(const RivalMachine *machine,
                                             RivalExecution *out,
                                             size_t capacity,
                                             size_t *out_len);

RivalError rival_machine_instruction_names(const RivalMachine *machine,
                                           const char *const **out_names,
                                           size_t *out_len);

uint32_t rival_machine_iterations(const RivalMachine *machine);

uint32_t rival_machine_bumps(const RivalMachine *machine);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  /* RIVAL3_FFI_H */
