"""Scalar autograd engine: the ``Value`` class.

Implements reverse-mode autodiff over a dynamically built DAG,
in the spirit of Karpathy's micrograd but with a slightly different API:

- ``Value`` carries a human-readable ``label`` (useful for graphviz plots)
- supports ``+ - * / ** tanh exp`` (+ ``relu`` for convenience)
- ``pow`` accepts int/float (a ``Value`` exponent is unwrapped to its data)
- ``zero_grad`` clears grads in the subgraph
"""

import math


class Value:
    """Stores a single scalar value and its gradient."""

    def __init__(self, data, _children=(), _op='', label='None'):
        self.data = data
        self.grad = 0.0
        self.label = label
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    # --- addition ---
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):  # other + self
        return self + other

    # --- multiplication ---
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other):  # other * self
        return self * other

    # --- power ---
    def __pow__(self, other):
        # unwrap Value exponents so `Value(2) ** Value(2)` keeps working
        # like in the original notebook; only the base gets a gradient
        # (same limitation as the notebook version).
        if isinstance(other, Value):
            other = other.data
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    # --- subtraction / division (defined via + * **, like the notebook) ---
    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), '-')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += -1.0 * out.grad
        out._backward = _backward

        return out

    def __rsub__(self, other):  # other - self
        other = other if isinstance(other, Value) else Value(other)
        return other - self

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        other = other if isinstance(other, Value) else Value(other)
        return other * self**-1

    # --- nonlinearities ---
    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    # --- backprop ---
    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        # clears grads of every node in the graph. call it BEFORE the next
        # accumulation buffer starts, i.e. right before optimizer.step(),
        # never between the backward() calls of one batch.
        # iterative version (handles shared subgraphs without revisiting).
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        for node in topo:
            node.grad = 0.0
