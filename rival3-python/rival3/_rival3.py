"""ctypes interface and Python expression compiler for the Rival C ABI."""

from __future__ import annotations

import ast
import ctypes as C
from enum import IntEnum
import math
import os
import platform
import re
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

ABI_VERSION = 3
INVALID_EXPR = 0xFFFFFFFF


class UnaryOp(IntEnum):
    NEG = 0
    FABS = 1
    SQRT = 2
    CBRT = 3
    POW2 = 4
    EXP = 5
    EXP2 = 6
    EXPM1 = 7
    LOG = 8
    LOG2 = 9
    LOG10 = 10
    LOG1P = 11
    LOGB = 12
    SIN = 13
    COS = 14
    TAN = 15
    ASIN = 16
    ACOS = 17
    ATAN = 18
    SINH = 19
    COSH = 20
    TANH = 21
    ASINH = 22
    ACOSH = 23
    ATANH = 24
    ERF = 25
    ERFC = 26
    LGAMMA = 27
    TGAMMA = 28
    RINT = 29
    ROUND = 30
    CEIL = 31
    FLOOR = 32
    TRUNC = 33
    NOT = 34
    ASSERT = 35
    ERROR = 36


class BinaryOp(IntEnum):
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3
    POW = 4
    HYPOT = 5
    FMIN = 6
    FMAX = 7
    FDIM = 8
    COPYSIGN = 9
    FMOD = 10
    REMAINDER = 11
    ATAN2 = 12
    AND = 13
    OR = 14
    EQ = 15
    NE = 16
    LT = 17
    LE = 18
    GT = 19
    GE = 20


class TernaryOp(IntEnum):
    FMA = 0
    IF = 1


class DiscType(IntEnum):
    BOOL = 0
    F32 = 1
    F64 = 2


class OutputPolicy(IntEnum):
    REQUIRE_ALL = 0
    ALLOW_PARTIAL = 1


class RivalError(RuntimeError):
    """Base class for errors returned by the native Rival library."""


class InvalidInputError(RivalError):
    """The expression is invalid for the supplied point."""


class UnsamplableError(RivalError):
    """Rival could not determine a result within the configured limits."""


class MisuseError(RivalError):
    """The native API rejected an invalid call or handle."""


class ABIError(ImportError):
    """The native library is missing or has an incompatible ABI."""


_ERRORS = {-1: InvalidInputError, -2: UnsamplableError, -3: MisuseError}


def _library_path() -> Path:
    system = platform.system()
    machine = platform.machine().lower()
    arch = {"amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}.get(machine, machine)
    if system == "Windows":
        subdir, filename = f"{arch}-windows", "rival3_ffi.dll"
    elif system == "Darwin":
        subdir, filename = f"{arch}-apple-darwin", "librival3_ffi.dylib"
    else:
        subdir, filename = f"{arch}-unknown-linux-gnu", "librival3_ffi.so"
    packaged = Path(__file__).parent / "private" / "native" / subdir / filename
    if packaged.is_file():
        return packaged
    env_path = os.environ.get("RIVAL3_LIBRARY")
    if env_path:
        return Path(env_path)
    raise ABIError(f"Rival native library not found for {system} {arch}; looked for {packaged}")


try:
    _lib = C.CDLL(str(_library_path()))
except OSError as exc:
    raise ABIError(f"Could not load Rival native library: {exc}") from exc


def _fn(name: str, restype, argtypes):
    fn = getattr(_lib, name)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


_voidp = C.c_void_p
_size = C.c_size_t
_u32 = C.c_uint32
_i32 = C.c_int32
_charp = C.c_char_p
_version = _fn("rival_version", _u32, [])
if _version() != ABI_VERSION:
    raise ABIError(f"Rival ABI mismatch: expected {ABI_VERSION}, got {_version()}")

_builder_new = _fn("rival_expr_builder_new", _i32, [C.POINTER(_charp), _size, C.POINTER(_voidp)])
_builder_free = _fn("rival_expr_builder_free", None, [_voidp])
_expr_var = _fn("rival_expr_var", _u32, [_voidp, _charp])
_expr_f64 = _fn("rival_expr_f64", _u32, [_voidp, C.c_double])
_expr_bigint = _fn("rival_expr_bigint", _u32, [_voidp, _charp])
_expr_bigrational = _fn("rival_expr_bigrational", _u32, [_voidp, _charp, _charp])
_expr_pi = _fn("rival_expr_pi", _u32, [_voidp])
_expr_e = _fn("rival_expr_e", _u32, [_voidp])
_expr_inf = _fn("rival_expr_infinity", _u32, [_voidp])
_expr_unary = _fn("rival_expr_unary", _u32, [_voidp, _u32, _u32])
_expr_binary = _fn("rival_expr_binary", _u32, [_voidp, _u32, _u32, _u32])
_expr_ternary = _fn("rival_expr_ternary", _u32, [_voidp, _u32, _u32, _u32, _u32])
_options_new = _fn("rival_machine_options_new", _i32, [C.POINTER(_voidp)])
_options_free = _fn("rival_machine_options_free", None, [_voidp])
_options_max_precision = _fn("rival_machine_options_set_max_precision", _i32, [_voidp, _u32])
_options_profile_capacity = _fn("rival_machine_options_set_profile_capacity", _i32, [_voidp, _size])
_machine_new = _fn("rival_machine_new", _i32, [_voidp, C.POINTER(_u32), C.POINTER(_u32), _size, _voidp, C.POINTER(_voidp)])
_machine_free = _fn("rival_machine_free", None, [_voidp])
_apply = _fn("rival_machine_apply_f64", _i32, [_voidp, C.POINTER(C.c_double), _size, _voidp, _u32, _u32, C.POINTER(C.c_double), _size])
_set_profiling = _fn("rival_machine_set_profiling", _i32, [_voidp, C.c_bool])
_get_profiling = _fn("rival_machine_get_profiling", C.c_bool, [_voidp])
_iterations = _fn("rival_machine_iterations", _u32, [_voidp])
_bumps = _fn("rival_machine_bumps", _u32, [_voidp])


def _check(status: int) -> None:
    if status:
        error = _ERRORS.get(status, RivalError)
        raise error(f"Rival call failed with status {status}")


_UNARY = {
    "neg": UnaryOp.NEG, "fabs": UnaryOp.FABS, "abs": UnaryOp.FABS,
    "sqrt": UnaryOp.SQRT, "cbrt": UnaryOp.CBRT, "pow2": UnaryOp.POW2,
    "exp": UnaryOp.EXP, "exp2": UnaryOp.EXP2, "expm1": UnaryOp.EXPM1,
    "log": UnaryOp.LOG, "log2": UnaryOp.LOG2, "log10": UnaryOp.LOG10,
    "log1p": UnaryOp.LOG1P, "logb": UnaryOp.LOGB, "sin": UnaryOp.SIN,
    "cos": UnaryOp.COS, "tan": UnaryOp.TAN, "asin": UnaryOp.ASIN,
    "acos": UnaryOp.ACOS, "atan": UnaryOp.ATAN, "sinh": UnaryOp.SINH,
    "cosh": UnaryOp.COSH, "tanh": UnaryOp.TANH, "asinh": UnaryOp.ASINH,
    "acosh": UnaryOp.ACOSH, "atanh": UnaryOp.ATANH, "erf": UnaryOp.ERF,
    "erfc": UnaryOp.ERFC, "lgamma": UnaryOp.LGAMMA, "tgamma": UnaryOp.TGAMMA,
    "rint": UnaryOp.RINT, "round": UnaryOp.ROUND, "ceil": UnaryOp.CEIL,
    "floor": UnaryOp.FLOOR, "trunc": UnaryOp.TRUNC, "not": UnaryOp.NOT,
    "assert": UnaryOp.ASSERT, "error": UnaryOp.ERROR,
}
_BINARY = {ast.Add: BinaryOp.ADD, ast.Sub: BinaryOp.SUB, ast.Mult: BinaryOp.MUL,
           ast.Div: BinaryOp.DIV, ast.Pow: BinaryOp.POW}
_CALLS = {name: (code, 1) for name, code in _UNARY.items()}
_CALLS.update({"pow": (BinaryOp.POW, 2), "hypot": (BinaryOp.HYPOT, 2),
               "fmin": (BinaryOp.FMIN, 2), "fmax": (BinaryOp.FMAX, 2),
               "fdim": (BinaryOp.FDIM, 2), "copysign": (BinaryOp.COPYSIGN, 2),
               "fmod": (BinaryOp.FMOD, 2), "remainder": (BinaryOp.REMAINDER, 2),
               "atan2": (BinaryOp.ATAN2, 2), "and": (BinaryOp.AND, 2),
               "or": (BinaryOp.OR, 2), "fma": (TernaryOp.FMA, 3)})
_COMPARE = {ast.Eq: BinaryOp.EQ, ast.NotEq: BinaryOp.NE, ast.Lt: BinaryOp.LT,
            ast.LtE: BinaryOp.LE, ast.Gt: BinaryOp.GT, ast.GtE: BinaryOp.GE}
_REAL_UNARY = {name for name, code in _UNARY.items() if code not in (UnaryOp.NOT, UnaryOp.ASSERT, UnaryOp.ERROR)}
_REAL_BINARY = {"pow", "hypot", "fmin", "fmax", "fdim", "copysign", "fmod", "remainder", "atan2"}


def _infer_type(node: ast.AST, variables: dict[str, Optional[str]], expected: Optional[str] = None) -> Optional[str]:
    """Infer types from operators and an optional declared output type."""
    def require(actual: Optional[str], wanted: str, what: str) -> str:
        if actual is not None and actual != wanted:
            raise TypeError(f"{what} must be {wanted}, got {actual}")
        return wanted

    if isinstance(node, ast.Constant):
        actual = "bool" if isinstance(node.value, bool) else "real" if isinstance(node.value, (int, float)) else None
        if actual is None:
            raise ValueError("Only numeric and boolean literals are supported")
        return require(actual, expected, "literal") if expected else actual
    if isinstance(node, ast.Name):
        if node.id in {"PI", "E", "INFINITY", "INF", "NAN"}:
            return require("real", expected, node.id) if expected else "real"
        if node.id not in variables:
            raise ValueError(f"Variable {node.id!r} was not declared")
        current = variables[node.id]
        if expected:
            variables[node.id] = require(current, expected, f"input {node.id!r}")
        return variables[node.id]
    if isinstance(node, ast.UnaryOp):
        wanted = "bool" if isinstance(node.op, ast.Not) else "real"
        if not isinstance(node.op, (ast.Not, ast.USub, ast.UAdd)):
            raise ValueError("Unsupported unary operator")
        require(expected, wanted, "unary expression") if expected else None
        _infer_type(node.operand, variables, wanted)
        return wanted
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.BitXor):
            wanted, operand_type = "bool", "bool"
        elif type(node.op) in _BINARY:
            wanted, operand_type = "real", "real"
        else:
            raise ValueError("Unsupported binary operator")
        require(expected, wanted, "binary expression") if expected else None
        _infer_type(node.left, variables, operand_type)
        _infer_type(node.right, variables, operand_type)
        return wanted
    if isinstance(node, ast.Compare):
        if not all(type(op) in _COMPARE for op in node.ops):
            raise ValueError("Unsupported comparison operator")
        require(expected, "bool", "comparison") if expected else None
        _infer_type(node.left, variables, "real")
        for operand in node.comparators:
            _infer_type(operand, variables, "real")
        return "bool"
    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, (ast.And, ast.Or)):
            raise ValueError("Unsupported boolean operator")
        require(expected, "bool", "boolean expression") if expected else None
        for value in node.values:
            _infer_type(value, variables, "bool")
        return "bool"
    if isinstance(node, ast.IfExp):
        _infer_type(node.test, variables, "bool")
        body_type = _infer_type(node.body, variables, expected)
        other_type = _infer_type(node.orelse, variables, expected or body_type)
        if body_type is None and other_type is not None:
            body_type = _infer_type(node.body, variables, other_type)
        if other_type is None and body_type is not None:
            other_type = _infer_type(node.orelse, variables, body_type)
        if body_type and other_type and body_type != other_type:
            raise TypeError("conditional branches must have the same type")
        return expected or body_type or other_type
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        name, args = node.func.id, node.args
        if name in _REAL_UNARY and len(args) == 1:
            wanted, arg_types = "real", ("real",)
        elif name in {"assert", "not"} and len(args) == 1:
            wanted, arg_types = "bool", ("bool",)
        elif name == "error" and len(args) == 1:
            wanted, arg_types = "bool", ("real",)
        elif name in _REAL_BINARY and len(args) == 2:
            wanted, arg_types = "real", ("real", "real")
        elif name in {"and", "or"} and len(args) == 2:
            wanted, arg_types = "bool", ("bool", "bool")
        elif name == "fma" and len(args) == 3:
            wanted, arg_types = "real", ("real", "real", "real")
        else:
            raise ValueError(f"Unsupported function or arity: {name}/{len(args)}")
        require(expected, wanted, name) if expected else None
        for arg, arg_type in zip(args, arg_types):
            _infer_type(arg, variables, arg_type)
        return wanted
    raise ValueError(f"Unsupported expression: {ast.dump(node, include_attributes=False)}")


def _lower(node: ast.AST, builder: int, source: str) -> int:
    # boolean literal
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return _expr_f64(builder, 1.0 if node.value else 0.0)
    # int/float literal
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        literal = ast.get_source_segment(source, node)
        if literal is None:
            raise ValueError("Could not recover numeric literal from expression")
        literal = literal.replace("_", "")
        if isinstance(node.value, int):
            exact = Fraction(node.value, 1)
        else:
            exact = Fraction(literal)
        return _expr_bigrational(builder, str(exact.numerator).encode(), str(exact.denominator).encode())
    if isinstance(node, ast.Name):
        constants = {"PI": _expr_pi, "E": _expr_e, "INF": _expr_inf}
        if node.id in constants:
            return constants[node.id](builder)
        if node.id == "NAN":
            return _expr_f64(builder, math.nan)
        return _expr_var(builder, node.id.encode())
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return _expr_unary(builder, int(UnaryOp.NEG), _lower(node.operand, builder, source))
        if isinstance(node.op, ast.Not):
            return _expr_unary(builder, int(UnaryOp.NOT), _lower(node.operand, builder, source))
        if isinstance(node.op, ast.UAdd):
            return _lower(node.operand, builder, source)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _expr_binary(builder, int(_BINARY[type(node.op)]), _lower(node.left, builder, source), _lower(node.right, builder, source))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitXor):
        return _expr_binary(builder, int(BinaryOp.NE), _lower(node.left, builder, source), _lower(node.right, builder, source))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        name = node.func.id
        if name in _CALLS:
            code, arity = _CALLS[name]
            if len(node.args) != arity:
                raise ValueError(f"{name} expects {arity} argument(s)")
            args = [_lower(arg, builder, source) for arg in node.args]
            if arity == 1:
                return _expr_unary(builder, int(code), args[0])
            return _expr_ternary(builder, int(code), *args) if arity == 3 else _expr_binary(builder, int(code), *args)
    if isinstance(node, ast.IfExp):
        return _expr_ternary(builder, int(TernaryOp.IF), _lower(node.test, builder, source), _lower(node.body, builder, source), _lower(node.orelse, builder, source))
    if isinstance(node, ast.Compare):
        result = None
        left = node.left
        for op, right in zip(node.ops, node.comparators):
            if type(op) not in _COMPARE:
                raise ValueError("Unsupported comparison operator")
            comparison = _expr_binary(builder, int(_COMPARE[type(op)]), _lower(left, builder, source), _lower(right, builder, source))
            result = comparison if result is None else _expr_binary(builder, int(BinaryOp.AND), result, comparison)
            left = right
        return result
    if isinstance(node, ast.BoolOp) and len(node.values) >= 2:
        code = BinaryOp.AND if isinstance(node.op, ast.And) else BinaryOp.OR
        result = _lower(node.values[0], builder, source)
        for value in node.values[1:]:
            result = _expr_binary(builder, int(code), result, _lower(value, builder, source))
        return result
    raise ValueError(f"Unsupported expression syntax: {ast.dump(node, include_attributes=False)}")


class Machine:
    """A compiled Rival expression set. Close it when finished."""

    def __init__(self, handle: int, variables: Sequence[str], input_types: Mapping[str, str], output_types: Sequence[str]):
        self._handle = handle
        self.variables = tuple(variables)
        self.input_types = dict(input_types)
        self.output_types = tuple(output_types)
        self.output_count = len(self.output_types)

    def close(self) -> None:
        if self._handle:
            _machine_free(self._handle)
            self._handle = None

    def __enter__(self):
        if not self._handle:
            raise RuntimeError("Machine is closed")
        return self

    def __exit__(self, *_exc):
        self.close()

    def apply(self, point: Mapping[str, object], *, max_iterations: int = 5, allow_partial: bool = False):
        if not self._handle:
            raise RuntimeError("Machine is closed")
        if set(point) != set(self.variables):
            raise ValueError(f"Expected values for exactly these variables: {self.variables}")
        converted = []
        for name in self.variables:
            value = point[name]
            if self.input_types[name] == "bool":
                if not isinstance(value, bool):
                    raise TypeError(f"Input {name!r} must be bool")
                converted.append(1.0 if value else 0.0)
            else:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"Input {name!r} must be int or float")
                converted.append(float(value))
        args = (C.c_double * len(self.variables))(*converted)
        out = (C.c_double * self.output_count)()
        policy = OutputPolicy.ALLOW_PARTIAL if allow_partial else OutputPolicy.REQUIRE_ALL
        _check(_apply(self._handle, args, len(args), None, max_iterations, int(policy), out, len(out)))
        values = tuple(bool(value) if typ == "bool" else value for value, typ in zip(out, self.output_types))
        return values[0] if self.output_count == 1 else values

    @property
    def profiling_enabled(self) -> bool:
        return bool(_get_profiling(self._handle))

    @profiling_enabled.setter
    def profiling_enabled(self, enabled: bool) -> None:
        _check(_set_profiling(self._handle, bool(enabled)))

    @property
    def iterations(self) -> int:
        return int(_iterations(self._handle))

    @property
    def bumps(self) -> int:
        return int(_bumps(self._handle))


def compile(expression: Union[str, Sequence[str]], variables: Sequence[str], *, discretizations: Union[str, Sequence[str]], max_precision: int = 10000, profile_capacity: int = 10000) -> Machine:
    """Compile expressions with required f64/bool output discretizations."""
    expressions = [expression] if isinstance(expression, str) else list(expression)
    variables = tuple(variables)
    if not expressions or len(set(variables)) != len(variables):
        raise ValueError("Provide at least one expression and unique variable names")
    if any(not isinstance(v, str) or not v.isidentifier() for v in variables):
        raise ValueError("Variable names must be Python identifiers")
    parsed = [ast.parse(re.sub(r"\bassert(?=\s*\()", "asrt__", expr), mode="eval").body for expr in expressions]
        
    # Python reserves `assert` as a statement, so string expressions use an
    # internal spelling that maps back to Rival's assert operator.
    for tree in parsed:
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "asrt__"):
                node.func.id = "assert"
    
    declared_outputs = [discretizations] if isinstance(discretizations, str) else list(discretizations)
    if len(declared_outputs) != len(parsed):
        raise ValueError("Provide one output type per expression")
    output_types = []
    for typ in declared_outputs:
        if typ == "f64":
            output_types.append("real")
        elif typ == "bool":
            output_types.append("bool")
        else:
            raise ValueError(f"Output type must be f64 or bool, got {typ!r}")
    inferred_inputs = {name: None for name in variables}
    for tree, declared in zip(parsed, output_types):
        inferred = _infer_type(tree, inferred_inputs, declared)
        if inferred != declared:
            raise TypeError(f"Expression output type is {inferred}, expected {declared}")
    unknown_inputs = [name for name, typ in inferred_inputs.items() if typ is None]
    if unknown_inputs:
        raise ValueError(f"Could not infer input type(s) for {unknown_inputs}; use them in a typed expression")
    names = (_charp * len(variables))(*(v.encode() for v in variables))
    builder = _voidp()
    _check(_builder_new(names, len(names), C.byref(builder)))
    options = _voidp()
    machine = _voidp()
    try:
        handles = [_lower(tree, builder, source) for tree, source in zip(parsed, expressions)]
        if any(handle == INVALID_EXPR for handle in handles):
            raise ValueError("Could not compile expression")
        _check(_options_new(C.byref(options)))
        _check(_options_max_precision(options, max_precision))
        _check(_options_profile_capacity(options, profile_capacity))
        handle_array = (_u32 * len(handles))(*handles)
        types = (_u32 * len(handles))(*(int(DiscType.BOOL if typ == "bool" else DiscType.F64) for typ in output_types))
        _check(_machine_new(builder, handle_array, types, len(handles), options, C.byref(machine)))
        return Machine(machine.value, variables, inferred_inputs, output_types)
    finally:
        if options:
            _options_free(options)
        _builder_free(builder)
