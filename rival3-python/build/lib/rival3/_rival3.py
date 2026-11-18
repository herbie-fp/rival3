"""ctypes interface and Python expression compiler for the Rival C ABI."""

from __future__ import annotations

import ast
import ctypes as C
from enum import IntEnum
import math
import os
import platform
import re
from decimal import Decimal, InvalidOperation
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
    INT = 3


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
    try:
        fn = getattr(_lib, name)
    except AttributeError as exc:
        raise ABIError(f"Native Rival library is missing required symbol {name}; rebuild rival3-ffi") from exc
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


_voidp = C.c_void_p
_size = C.c_size_t
_u32 = C.c_uint32
_i32 = C.c_int32
_charp = C.c_char_p


class _TextInput(C.Structure):
    _fields_ = [("kind", _u32), ("value", _charp)]


class _Execution(C.Structure):
    _fields_ = [("instruction_idx", _i32), ("precision", _u32),
                ("time_ms", C.c_double), ("iteration", _u32)]
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
# This is faster version of apply when working strictly with floats
_apply_f64 = _fn("rival_machine_apply_f64", _i32, [_voidp, C.POINTER(C.c_double), _size, _voidp, _u32, _u32, C.POINTER(C.c_double), _size])
# This is a slower version of apply when passing inputs as strings
_apply_text = _fn("rival_machine_apply_text", _i32, [_voidp, C.POINTER(_TextInput), _size, _voidp, _u32, _u32, C.POINTER(_voidp), _size])
_string_free = _fn("rival_string_free", None, [_voidp])
_set_profiling = _fn("rival_machine_set_profiling", _i32, [_voidp, C.c_bool])
_get_profiling = _fn("rival_machine_get_profiling", C.c_bool, [_voidp])
_profiler_reset = _fn("rival_machine_profiler_reset", _i32, [_voidp])
_profiler_executions = _fn("rival_machine_profiler_executions", _i32, [_voidp, C.POINTER(_Execution), _size, C.POINTER(_size)])
_instruction_names = _fn("rival_machine_instruction_names", _i32, [_voidp, C.POINTER(C.POINTER(_charp)), C.POINTER(_size)])
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
    "error": UnaryOp.ERROR,
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
_REAL_UNARY = {name for name, code in _UNARY.items() if code not in (UnaryOp.NOT, UnaryOp.ERROR)}
_REAL_BINARY = {"pow", "hypot", "fmin", "fmax", "fdim", "copysign", "fmod", "remainder", "atan2"}


# This function tries infer type of a variable given expected output type of an expression
def _infer_type(node: ast.AST, variables: dict[str, Optional[str]], expected: str) -> str:
    
    def require(actual: Optional[str], wanted: str, what: str) -> str:
        if actual != wanted:
            raise TypeError(f"{what} must be {wanted}, got {actual}")
        return wanted

    if isinstance(node, ast.Constant):
        actual = "bool" if isinstance(node.value, bool) else "real" if isinstance(node.value, (int, float)) else None
        if actual is None:
            raise ValueError("Only numeric and boolean literals are supported")
        return require(actual, expected, f"literal {node.value}")
    if isinstance(node, ast.Name):
        if node.id in {"PI", "E", "INF", "NAN"}:
            return require("real", expected, node.id)
        if node.id not in variables:
            raise ValueError(f"Variable {node.id!r} was not declared")
        if variables[node.id] is None:
            variables[node.id] = variables[node.id]
        else:
            require(variables[node.id], expected, f"input {node.id!r}")
        return variables[node.id]
    if isinstance(node, ast.UnaryOp):
        wanted = "bool" if isinstance(node.op, ast.Not) else "real"
        if not isinstance(node.op, (ast.Not, ast.USub, ast.UAdd)):
            raise ValueError("Unsupported unary operator")
        require(expected, wanted, "unary expression")
        _infer_type(node.operand, variables, wanted)
        return wanted
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.BitXor):
            wanted, operand_type = "bool", "bool"
        elif type(node.op) in _BINARY:
            wanted, operand_type = "real", "real"
        else:
            raise ValueError("Unsupported binary operator")
        require(expected, wanted, "binary expression")
        _infer_type(node.left, variables, operand_type)
        _infer_type(node.right, variables, operand_type)
        return wanted
    if isinstance(node, ast.Compare):
        if not all(type(op) in _COMPARE for op in node.ops):
            raise ValueError("Unsupported comparison operator")
        require(expected, "bool", "comparison")
        _infer_type(node.left, variables, "real")
        for operand in node.comparators:
            _infer_type(operand, variables, "real")
        return "bool"
    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, (ast.And, ast.Or)):
            raise ValueError("Unsupported boolean operator")
        require(expected, "bool", "boolean expression")
        _infer_type(node.left, variables, "bool")
        _infer_type(node.right, variables, "bool")
        return "bool"
    if isinstance(node, ast.IfExp):
        _infer_type(node.test, variables, "bool")
        _infer_type(node.body, variables, expected)
        _infer_type(node.orelse, variables, expected)
        return expected
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        name, args = node.func.id, node.args
        if name in _REAL_UNARY and len(args) == 1:
            wanted, arg_types = "real", ("real",)
        elif name == "not" and len(args) == 1:
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
        require(expected, wanted, name)
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


def _rational(text: str) -> Fraction:
    """Parse finite decimal/scientific notation or an integer fraction exactly."""
    try:
        if "/" in text:
            parts = text.split("/")
            if len(parts) != 2 or any(re.fullmatch(r"[+-]?[0-9]+", part.strip()) is None for part in parts):
                raise ValueError("Expected an integer numerator and denominator")
            # Decimal construction avoids Python's decimal-string digit limit.
            return Fraction(*(int(Decimal(part.strip())) for part in parts))
        value = Decimal(text)
        if not value.is_finite():
            raise ValueError("Rational inputs must be finite")
        return Fraction(value)
    except (InvalidOperation, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Invalid exact rational input: {text!r}") from exc


def _encode_input(value: object, expected: Optional[str], name: str) -> tuple[int, bytes]:
    if expected == "bool" and not isinstance(value, bool):
        raise TypeError(f"Input {name!r} must be bool")
    if isinstance(value, bool):
        if expected == "real":
            raise TypeError(f"Input {name!r} must be int, float, or an exact rational string")
        return 3, b"1" if value else b"0"
    if isinstance(value, int):
        return 0, str(Decimal(value)).encode("ascii")
    if isinstance(value, float):
        return 1, repr(value).encode("ascii")
    if isinstance(value, str):
        rational = _rational(value)
        return 2, f"{Decimal(rational.numerator)}/{Decimal(rational.denominator)}".encode("ascii")
    raise TypeError(f"Input {name!r} must be int, float, bool, or an exact rational string")


def _decode_output(text: str, kind: str):
    if kind == "int":
        return int(Decimal(text))
    if kind == "bool":
        return text == "1"
    if kind == "float":
        return float(text)
    else:
        raise NotImplementedError()


class Machine:
    """A compiled Rival expression set. Close it when finished."""

    def __init__(self, handle: int, variables: Sequence[str], input_types: Mapping[str, Optional[str]], discs: Sequence[str]):
        self._handle = handle
        self.variables = tuple(variables)
        self.input_types = dict(input_types)
        self.discs = tuple(discs)
        self.output_count = len(discs)

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

    def apply(self, *point: object):
        max_iterations = 5
        # Require all outputs to be evaluated no matter errors we get
        policy = OutputPolicy.REQUIRE_ALL

        if not self._handle:
            raise RuntimeError("Machine is closed")
        if len(point) != len(self.variables):
            raise ValueError(f"Expected {len(self.variables)} input value(s) in variable order {self.variables}")

        # Fast float-pass
        if all(type(value) is float for value in point) and all(disc != "int" for disc in self.discs):
            args = (C.c_double * len(point))(*point)
            out = (C.c_double * self.output_count)()
            _check(_apply_f64(self._handle, args, len(args), None, max_iterations, int(policy), out, len(out)))
            values = tuple(bool(value) if disc == "bool" else value for value, disc in zip(out, self.discs))
        
        # Slower pass for exactly representing rationals/integers in Rust
        else:
            encoded = [
                _encode_input(value, self.input_types[name], name)
                for name, value in zip(self.variables, point)
            ]
            args = (_TextInput * len(encoded))(*(_TextInput(kind, value) for kind, value in encoded))
            out = (_voidp * self.output_count)()
            try:
                _check(_apply_text(self._handle, args, len(args), None, max_iterations, int(policy), out, len(out)))
                values = tuple(
                    _decode_output(C.string_at(ptr).decode("ascii"), typ) if ptr else None
                    for ptr, typ in zip(out, self.discs)
                )
            finally:
                for ptr in out:
                    if ptr:
                        _string_free(ptr)
        return values[0] if self.output_count == 1 else values

    @property
    def profiling_enabled(self) -> bool:
        return bool(_get_profiling(self._handle))

    @profiling_enabled.setter
    def profiling_enabled(self, enabled: bool) -> None:
        _check(_set_profiling(self._handle, bool(enabled)))

    @property
    def profiling_info(self) -> list[dict[str, object]]:
        """Return a snapshot of recorded instruction executions."""
        names_ptr = C.POINTER(_charp)()
        name_count = _size()
        _check(_instruction_names(self._handle, C.byref(names_ptr), C.byref(name_count)))
        names = [names_ptr[i].decode("utf-8") for i in range(name_count.value)]

        count = _size()
        _check(_profiler_executions(self._handle, None, 0, C.byref(count)))
        records = (_Execution * count.value)()
        _check(_profiler_executions(self._handle, records, len(records), C.byref(count)))
        return [
            {"instruction": names[record.instruction_idx],
             "precision": record.precision,
             "time_ms": record.time_ms,
             "iteration": record.iteration}
            for record in records[:count.value]
        ]

    def reset_profiling(self) -> None:
        """Clear recorded instruction executions."""
        _check(_profiler_reset(self._handle))

    


def compile(expressions: list[str], variables: list[str], *, discs: list[str], max_precision: int = 10000, profile_capacity: int = 10000) -> Machine:
    """Compile expression and variable lists with required output discretizations."""
    if not isinstance(expressions, list) or not isinstance(variables, list) or not isinstance(discs, list):
        raise TypeError("expressions, variables, and discs must be lists")
    if any(not isinstance(expr, str) for expr in expressions):
        raise TypeError("every expression must be a string")
    variables = tuple(variables)
    if not expressions or len(set(variables)) != len(variables):
        raise ValueError("Provide at least one expression and unique variable names")
    if any(not isinstance(v, str) or not v.isidentifier() for v in variables):
        raise ValueError("Variable names must be Python identifiers")
    parsed = [ast.parse(expr, mode="eval").body for expr in expressions]

    if len(discs) != len(parsed):
        raise ValueError("Provide one output type per expression")

    disc_types = {"bool": DiscType.BOOL, "float": DiscType.F64, "int": DiscType.INT}
    for disc in discs:
        if disc not in disc_types:
            raise ValueError(f"Discs must be float, bool or int, got {disc!r}")

    # Little typecheck and inferring input types
    otypes = ["bool" if disc == "bool" else "real" for disc in discs]
    itypes = {name: None for name in variables}
    for tree, otype in zip(parsed, otypes):
        _infer_type(tree, itypes, otype)
        
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
        
        types = (_u32 * len(handles))(*(int(disc_types[disc]) for disc in discs))
        
        _check(_machine_new(builder, handle_array, types, len(handles), options, C.byref(machine)))
        return Machine(machine.value, variables, itypes, discs)
    finally:
        if options:
            _options_free(options)
        _builder_free(builder)
