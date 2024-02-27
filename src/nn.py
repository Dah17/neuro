from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, List
from src.value import Value
from abc import ABC
import random as rn


class Module(ABC):
    """
    Base class for all sub module
    """

    def zero_grad(self):
        """ """
        for param in self.parameters:
            param.zero_grad()

    @property
    def parameters(self):
        """ """
        return []


class Neuron(Module):
    """ """

    def __init__(self, nin: int, nonlin: bool = True, **kwargs):
        """ """
        self.bias = Value(0)
        self.nonlin = nonlin
        self.weights = [Value(rn.uniform(-1, 1)) for _ in range(nin)]

    def __call__(self, x: List[Value]):
        out = sum([wi * xi for wi, xi in zip(self.weights, x)], self.bias)
        return out.relu() if self.nonlin else out

    @property
    def parameters(self):
        """ """
        return self.weights + [self.bias]

    def __repr__(self) -> str:
        return f"{ 'Relu' if self.nonlin else 'Linear'}Neuron({len(self.weights)})"


class Layer(Module):
    """ """

    def __init__(self, nin: int, nout: int, nonlin: bool = True, **kwargs):
        """ """
        self.nout = nout
        self.neurons = [Neuron(nin, nonlin=nonlin, **kwargs) for _ in range(nout)]

    def __call__(self, X: Value | List[Value]):
        """ """
        X = [X] if isinstance(X, Value) else X
        out = [n(X) for n in self.neurons]
        return out[0] if self.nout == 1 else out

    @property
    def parameters(self):
        """ """
        return [p for n in self.neurons for p in n.parameters]

    def __repr__(self) -> str:
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """Module"""

    def __init__(self, nin: int, nouts: List[int]):
        """ """
        sz = [nin] + nouts
        self.nouts = nouts
        self.layers = [
            Layer(nin=sz[i], nout=sz[i + 1], nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def __call__(self, X: Value | List[Value]):
        """ """
        X = [X] if isinstance(X, Value) else X
        for layer in self.layers:
            X = layer(X)
        return X

    @property
    def parameters(self):
        """ """
        return [p for l in self.layers for p in l.parameters]

    def __repr__(self) -> str:
        return f"MLP of [{', '.join(str(l) for l in self.layers)}]"
