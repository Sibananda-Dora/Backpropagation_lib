"""Tests for the Value autograd engine (numerical-grad checked, no torch needed)."""

import math
import random

from backprop.engine import Value
from backprop.nn import Neuron, Layer, MLP


def numerical_grad(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)


def test_add_mul_backward():
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a * b
    d = e + c
    f = Value(-2.0, label='f')
    L = d * f
    L.backward()
    # L = ((a*b) + c) * f  =>  dL/da = b*f, dL/db = a*f, dL/dc = f, etc.
    assert L.data == -8.0
    assert a.grad == (-3.0) * (-2.0)
    assert b.grad == 2.0 * (-2.0)
    assert c.grad == -2.0
    assert e.grad == -2.0
    assert d.grad == -2.0
    assert f.grad == 4.0


def test_sub_div_pow():
    a = Value(3.0)
    b = Value(2.0)
    c = a - b  # 1
    d = a / b  # 1.5
    e = a ** 2  # 9
    c.backward()
    assert c.data == 1.0
    assert a.grad == 1.0 and b.grad == -1.0

    a.grad = b.grad = 0.0
    d.backward()
    assert d.data == 1.5
    assert abs(a.grad - 1 / 2.0) < 1e-9
    assert abs(b.grad - (-3.0 / 4.0)) < 1e-9

    a.grad = 0.0
    e.backward()
    assert e.data == 9.0
    assert a.grad == 6.0


def test_tanh_exp_vs_numerical():
    for fn_name in ("tanh", "exp"):
        x0 = 0.8813735870195432
        x = Value(x0)
        y = getattr(x, fn_name)()
        y.backward()
        if fn_name == "tanh":
            f = math.tanh
        else:
            f = math.exp
        expected = numerical_grad(f, x0)
        assert abs(y.data - f(x0)) < 1e-9
        assert abs(x.grad - expected) < 1e-5


def test_neuron_mlp_forward_and_zero_grad():
    random.seed(0)
    n = MLP(3, [4, 4, 1])
    # 3-in, 4+4+1-out MLP: (3*4+4) + (4*4+4) + (4*1+1) = 41 params
    assert len(n.parameters()) == 41
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5]]
    ypred = [n(x) for x in xs]
    loss = sum((y - 1.0) ** 2 for y in ypred)
    loss.backward()
    assert any(p.grad != 0.0 for p in n.parameters())
    n.zero_grad()
    assert all(p.grad == 0.0 for p in n.parameters())


def test_broadcast_with_scalars():
    a = Value(2.0)
    b = a + 1
    c = 3 * a
    d = a - 5
    assert (b.data, c.data, d.data) == (3.0, 6.0, -3.0)
    b.backward()
    assert a.grad == 1.0
