"""ctypes interface and Python expression compiler for the Rival C ABI."""

from __future__ import annotations

import ast
import ctypes as C
import math
import os
import platform
import re
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence, Union

ABI_VERSION = 3
INVALID_EXPR = 0xFFFFFFFF


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
    "neg": 0, "fabs": 1, "abs": 1, "sqrt": 2, "cbrt": 3, "pow2": 4, "exp": 5, "exp2": 6,
    "expm1": 7, "log": 8, "log2": 9, "log10": 10, "log1p": 11,
    "logb": 12, "sin": 13, "cos": 14, "tan": 15, "asin": 16,
    "acos": 17, "atan": 18, "sinh": 19, "cosh": 20, "tanh": 21,
    "asinh": 22, "acosh": 23, "atanh": 24, "erf": 25, "erfc": 26,
    "lgamma": 27, "tgamma": 28, "rint": 29, "round": 30, "ceil": 31,
    "floor": 32, "trunc": 33, "not": 34, "assert": 35, "error": 36,
}
_BINARY = {ast.Add: 0, ast.Sub: 1, ast.Mult: 2, ast.Div: 3, ast.Pow: 4}
_CALLS = {name: (code, 1) for name, code in _UNARY.items()}
_CALLS.update({"pow": (4, 2), "hypot": (5, 2), "fmin": (6, 2), "fmax": (7, 2),
               "fdim": (8, 2), "copysign": (9, 2), "fmod": (10, 2),
               "remainder": (11, 2), "atan2": (12, 2), "and": (13, 2),
               "or": (14, 2), "fma": (0, 3)})
_COMPARE = {ast.Eq: 15, ast.NotEq: 16, ast.Lt: 17, ast.LtE: 18, ast.Gt: 19, ast.GtE: 20}
_REAL_UNARY = {name for name, code in _UNARY.items() if code not in (34, 35, 36)}
_REAL_BINARY = {"pow", "hypot", "fmin", "fmax", "fdim", "copysign", "fmod", "remainder", "atan2"}


def _type_of(node: ast.AST) -> str:
    """Infer Rival's real/bool type and reject ill-typed expression trees."""
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return "bool"
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return "real"
    if isinstance(node, ast.Name):
        if node.id in {"PI", "E", "INFINITY", "NAN"}:
            return "real"
        return "real"  # Machine inputs are f64 values in v1.
    if isinstance(node, ast.UnaryOp):
        typ = _type_of(node.operand)
        if isinstance(node.op, ast.Not):
            if typ != "bool": raise ValueError("not requires a boolean expression")
            return "bool"
        if isinstance(node.op, (ast.USub, ast.UAdd)) and typ == "real":
            return "real"
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        if _type_of(node.left) == _type_of(node.right) == "real":
            return "real"
    if isinstance(node, ast.Compare):
        operands = [node.left, *node.comparators]
        if all(_type_of(x) == "real" for x in operands) and all(type(op) in _COMPARE for op in node.ops):
            return "bool"
    if isinstance(node, ast.BoolOp) and all(_type_of(x) == "bool" for x in node.values):
        return "bool"
    if isinstance(node, ast.IfExp):
        if _type_of(node.test) == "bool":
            body, other = _type_of(node.body), _type_of(node.orelse)
            if body == other:
                return body
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        name = node.func.id
        args = node.args
        if name in _REAL_UNARY and len(args) == 1 and _type_of(args[0]) == "real": return "real"
        if name in {"assert", "not"} and len(args) == 1 and _type_of(args[0]) == "bool": return "bool"
        if name == "error" and len(args) == 1 and _type_of(args[0]) == "real": return "bool"
        if name in _REAL_BINARY and len(args) == 2 and all(_type_of(x) == "real" for x in args): return "real"
        if name in {"and", "or"} and len(args) == 2 and all(_type_of(x) == "bool" for x in args): return "bool"
        if name == "fma" and len(args) == 3 and all(_type_of(x) == "real" for x in args): return "real"
    raise ValueError(f"Unsupported or ill-typed expression: {ast.dump(node, include_attributes=False)}")


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
            return _expr_unary(builder, 0, _lower(node.operand, builder, source))
        if isinstance(node.op, ast.Not):
            return _expr_unary(builder, 34, _lower(node.operand, builder, source))
        if isinstance(node.op, ast.UAdd):
            return _lower(node.operand, builder, source)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _expr_binary(builder, _BINARY[type(node.op)], _lower(node.left, builder, source), _lower(node.right, builder, source))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        name = node.func.id
        if name in _CALLS:
            code, arity = _CALLS[name]
            if len(node.args) != arity:
                raise ValueError(f"{name} expects {arity} argument(s)")
            args = [_lower(arg, builder, source) for arg in node.args]
            if arity == 1:
                return _expr_unary(builder, code, args[0])
            return _expr_ternary(builder, 0, *args) if arity == 3 else _expr_binary(builder, code, *args)
    if isinstance(node, ast.IfExp):
        return _expr_ternary(builder, 1, _lower(node.test, builder, source), _lower(node.body, builder, source), _lower(node.orelse, builder, source))
    if isinstance(node, ast.Compare):
        result = None
        left = node.left
        for op, right in zip(node.ops, node.comparators):
            if type(op) not in _COMPARE:
                raise ValueError("Unsupported comparison operator")
            comparison = _expr_binary(builder, _COMPARE[type(op)], _lower(left, builder, source), _lower(right, builder, source))
            result = comparison if result is None else _expr_binary(builder, 13, result, comparison)
            left = right
        return result
    if isinstance(node, ast.BoolOp) and len(node.values) >= 2:
        code = 13 if isinstance(node.op, ast.And) else 14
        result = _lower(node.values[0], builder, source)
        for value in node.values[1:]:
            result = _expr_binary(builder, code, result, _lower(value, builder, source))
        return result
    raise ValueError(f"Unsupported expression syntax: {ast.dump(node, include_attributes=False)}")


class Machine:
    """A compiled Rival expression set. Close it when finished."""

    def __init__(self, handle: int, variables: Sequence[str], output_types: Sequence[str]):
        self._handle = handle
        self.variables = tuple(variables)
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

    def apply(self, point: Mapping[str, float], *, max_iterations: int = 5, allow_partial: bool = False):
        if not self._handle:
            raise RuntimeError("Machine is closed")
        if set(point) != set(self.variables):
            raise ValueError(f"Expected values for exactly these variables: {self.variables}")
        args = (C.c_double * len(self.variables))(*(float(point[v]) for v in self.variables))
        out = (C.c_double * self.output_count)()
        _check(_apply(self._handle, args, len(args), None, max_iterations, int(allow_partial), out, len(out)))
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


def compile(expression: Union[str, Sequence[str]], variables: Sequence[str], *, max_precision: int = 10000, profile_capacity: int = 10000) -> Machine:
    """Compile one expression or a sequence of expressions using f64 outputs."""
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
    output_types = [_type_of(tree) for tree in parsed]
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
        types = (_u32 * len(handles))(*(0 if typ == "bool" else 2 for typ in output_types))
        _check(_machine_new(builder, handle_array, types, len(handles), options, C.byref(machine)))
        return Machine(machine.value, variables, output_types)
    finally:
        if options:
            _options_free(options)
        _builder_free(builder)
