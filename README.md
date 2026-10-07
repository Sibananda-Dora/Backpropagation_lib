# backprop-lib

A tiny scalar-valued autograd engine and a neural net library on top of it with a PyTorch-like API — in the spirit of [Karpathy's micrograd](https://github.com/karpathy/micrograd), but with a slightly different approach (see Differences).

Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG, plus `Neuron` / `Layer` / `MLP` on top. Extracted from `manual_backprop.ipynb` into an installable package:

```
backprop/
  __init__.py   # package entry: Value + nn
  engine.py     # Value autograd engine (+ - * / ** tanh exp relu)
  nn.py         # Module, Neuron, Layer, MLP (tanh-based)
  viz.py        # trace() + draw_dot() graphviz helpers (optional dep)
test/
  test_engine.py
manual_backprop.ipynb  # original derivation / demo notebook
```

## Installation

From PyPI (after you publish — see below):

```bash
pip install backprop-lib
```

From source:

```bash
pip install -e ".[viz]"
```

With dev/test tools:

```bash
pip install -e ".[dev]"
pytest
```

## Example usage

```python
from backprop.engine import Value
from backprop import nn

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'

L.backward()
print(a.grad, b.grad)  # 6.0 -4.0
```

Training a small MLP (same as the notebook):

```python
from backprop.nn import MLP

xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]
model = MLP(3, [4, 4, 1])

for k in range(20):
    ypred = [model(x) for x in xs]
    loss = sum((y - ygt) ** 2 for y, ygt in zip(ypred, ys))
    model.zero_grad()
    loss.backward()
    for p in model.parameters():
        p.data += -0.1 * p.grad
    print(k, loss.data)
```

Graphviz visualisation (needs `pip install backprop-lib[viz]` + Graphviz binary):

```python
from backprop.viz import draw_dot
draw_dot(L)
```

Note: the `graphviz` pip package does not install the Graphviz system binary and
does not set `PATH` for you. On Windows, install from graphviz.org with `Add to PATH`
checked (or `winget install graphviz`), otherwise add `C:\Program Files\Graphviz\bin`
to `PATH` manually, restart the terminal, and verify with `dot -V`.

## Differences from micrograd

- `Value(..., label='...')` — every node carries a debug label used by `draw_dot`.
- Activations: `tanh` + `exp` are core (matching the notebook's manual neuron derivation); `relu` is also included for convenience. Micrograd's `nn` defaults to ReLU; this `nn` uses `tanh` everywhere and randomises the bias.
- `Value.zero_grad()` clears the whole subgraph (iterative, sharing-safe); `Module.zero_grad()` clears parameters.
- `__pow__` unwraps a `Value` exponent to its `.data` so notebook code like `other**Value(-1.0)` keeps working; plain int/float exponents are preferred.
- Explicit `__sub__` backward (`+1/-1`) instead of going through `__neg__`.

## Publishing to PyPI

1. Rename later if you want: edit `name = "backprop-lib"` in `pyproject.toml`
   (and the `backprop/` folder if you want the import name to change too),
   then update imports + this README.
2. Bump `version`.
3. Build + upload:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine upload dist/*
# TestPyPI first (recommended):
# python -m twine upload --repository testpypi dist/*
```

## Running tests

```bash
pytest -v
```

Tests use only the stdlib (`math`, `random`) + `pytest` — no `torch` required.

## License

MIT
