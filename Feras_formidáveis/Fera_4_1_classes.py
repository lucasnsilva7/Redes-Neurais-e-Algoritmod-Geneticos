import random
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class Valor:
    def __init__(self, data, parents=(), op=""):
        self.data = data
        self.parents = parents
        self.op = op
        self.grad = 0.0
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Valor) else Valor(other)
        out = Valor(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Valor) else Valor(other)
        out = Valor(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Valor(e, (self,), "exp")
        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self): # Fera 4.1
        eps = 1e-15
        x = max(self.data, eps)
        out = Valor(math.log(x), (self,), "log")
        def _backward():
            self.grad += (1.0 / x) * out.grad
        out._backward = _backward
        return out

    def __pow__(self, power):
        out = Valor(self.data**power, (self,), f"**{power}")
        def _backward():
            self.grad += power * (self.data**(power-1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1

    def relu(self): # Fera 4.1
        out = Valor(self.data if self.data > 0 else 0.0, (self,), "relu")
        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def sig(self):
        e = self.exp()
        return e / (e + Valor(1.0))

    def backward(self):
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for p in v.parents:
                    build(p)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

class Neuronio:
    def __init__(self, num_entradas):
        std = math.sqrt(2.0 / num_entradas)
        self.bias = Valor(0.0)
        self.pesos = [Valor(random.gauss(0.0, std)) for _ in range(num_entradas)]

    def __call__(self, entrada, use_relu=True):
        soma = Valor(0.0)
        for x, w in zip(entrada, self.pesos):
            soma = soma + x * w
        soma = soma + self.bias
        return soma.relu() if use_relu else soma.sig()

    def params(self):
        return self.pesos + [self.bias]

class Camada:
    def __init__(self, num_neuronios, num_entradas):
        self.neuronios = [Neuronio(num_entradas) for _ in range(num_neuronios)]

    def __call__(self, entrada, use_relu=True):
        saidas = [neuronio(entrada, use_relu) for neuronio in self.neuronios]
        return saidas[0] if len(saidas) == 1 else saidas

    def params(self):
        return [p for neuronio in self.neuronios for p in neuronio.params()]

class MLP:
    def __init__(self, num_entradas, tamanhos_ocultos):
        tamanhos = [num_entradas] + tamanhos_ocultos
        self.camadas = [Camada(tamanhos[i+1], tamanhos[i]) for i in range(len(tamanhos_ocultos))]

    def __call__(self, entrada):
        for i, camada in enumerate(self.camadas):
            use_relu = (i < len(self.camadas) - 1)
            entrada = camada(entrada, use_relu)
        return entrada

    def params(self):
        return [p for camada in self.camadas for p in camada.params()]

def cross_entropy(saidas_reais, saidas_preditas): # Fera 4.1
    N = len(saidas_reais)
    soma = Valor(0.0)
    for y_true, y_pred in zip(saidas_reais, saidas_preditas):
        t1 = Valor(y_true) * y_pred.log()
        t2 = (Valor(1.0) - Valor(y_true)) * (Valor(1.0) - y_pred).log()
        soma = soma + (t1 + t2)
    return soma * Valor(-1.0 / N)