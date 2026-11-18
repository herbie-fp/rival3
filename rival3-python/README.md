# Rival 3 for Python

Python bindings to Rival 3. Expressions are written as Python-like strings and
compiled to a reusable machine:

```python
from rival3 import compile

machine = compile(["x * y + sin(z)"], ["x", "y", "z"], discs=["float"])
try:
    result = machine.apply(2.0, 3.0, 0.5)
finally:
    machine.close()
```

V1 evaluates f64 inputs. Analysis, hints, and arbitrary-precision inputs are
not yet exposed.

Input types are inferred from the expression. For a bare variable, declare its
output type so the compiler can infer the corresponding input type:

```python
with compile(["enabled"], ["enabled"], discs=["bool"]) as machine:
    result = machine.apply(True)
```

Real inputs accept Python integers or floats; boolean inputs require `bool`.
