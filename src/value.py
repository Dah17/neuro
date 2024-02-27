from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union, Optional, List, Callable
import math


@dataclass
class Value:
    """
    Object that proivde efficient calculation mechanism
    for neural network operations
    """

    data: Union[int, float]
    label: str = ""
    grad: float = 0.0
    inner_backward: Callable[[], None] = lambda: None
    operation: Optional[str] = None
    children: List[Value] = field(default_factory=lambda: [])

    def __post_init__(self):
        children = [child for child in set(self.children)]

    def __hash__(self) -> int:
        return hash(str(self))

    def __repr__(self) -> str:
        """Return"""
        return f"Value(data={self.data},grad={self.grad})"

    def zero_grad(self):
        """ """
        self.grad = 0.0

    def backward(self):
        """ """
        topo: List[Value] = []
        visited = set()

        def buid_topo(v: Value):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    buid_topo(child)
                topo.append(v)

        buid_topo(self)

        self.grad = 1
        for x in reversed(topo):
            x.inner_backward()

    def exp(self) -> Value:
        """ """
        out = Value(math.exp(self.data), operation="exp", children=[self])

        def backward():
            self.grad += math.exp(self.data) * out.grad

        out.inner_backward = backward

        return out

    def tanh(self) -> Value:
        """ """
        return ((self * 2).exp() - 1) / ((self * 2).exp() + 1)

    def relu(self) -> Value:
        """ """
        out = Value(max(0, self.data), operation="Relu", children=[self])

        def backward():
            self.grad += (self.data > 0) * out.grad

        out.inner_backward = backward

        return out

    def __add__(self, other: Union[Value, int, float]) -> Value:
        """
        Add another value to this
        """

        other = other if isinstance(other, Value) else Value(other, label=str(other))
        out = Value(self.data + other.data, operation="+", children=[self, other])

        def backward():
            self.grad += out.grad
            other.grad += out.grad

        out.inner_backward = backward

        return out

    def __mul__(self, other: Union[Value, int, float]) -> Value:
        """
        Add another value to this
        """
        other = other if isinstance(other, Value) else Value(other, label=str(other))
        out = Value(self.data * other.data, operation="*", children=[other, self])

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out.inner_backward = backward

        return out

    def __pow__(self, other: Union[int, float]) -> Value:
        """
        Pow
        """
        out = Value(self.data**other, operation="**", children=[self])

        def backward():
            self.grad += (other * (self.data**other - 1)) * out.grad

        out.inner_backward = backward

        return out

    def __truediv__(self, other: Union[Value, int, float]) -> Value:
        """
        Add another value to this
        """
        other = other if isinstance(other, Value) else Value(other, label=str(other))
        return self * other**-1

    def __rtruediv__(self, other: Union[Value, int, float]) -> Value:
        """ """
        other = other if isinstance(other, Value) else Value(other, label=str(other))
        return other * self**-1

    def __radd__(self, other: Union[Value, int, float]) -> Value:
        """
        Add another value to this
        """
        return self + other

    def __rsub__(self, other: Union[Value, int, float]) -> Value:
        """
        Add another value to this
        """
        other = other if isinstance(other, Value) else Value(other, label=str(other))
        return other - self

    def __rmul__(self, other: Union[Value, int, float]) -> Value:
        """
        Add another value to this
        """
        return self * other

    def __neg__(self) -> Value:
        """ """
        return self * (-1)

    def __sub__(self, other: Union[Value, int, float]) -> Value:
        """
        Add another value to this
        """
        return self + (-other)
