"""Implicit field definitions and density calibration for direct volume meshing.

The trigonometric functions are level-set approximations, not exact minimal
surfaces. Density grading uses the CDF of a sampled periodic unit cell.
"""

from __future__ import annotations
import ast
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
from pathlib import Path
import operator
import numpy as np

DATA = Path(__file__).with_name("families.json")
FAMILIES = json.loads(DATA.read_text(encoding="utf-8"))
FUNCTIONS = {name: getattr(np, name) for name in ("sin", "cos", "tan", "sqrt", "abs", "exp")}
OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


@lru_cache(maxsize=128)
def parse_expression(expression):
    if not isinstance(expression, str) or len(expression) > 2000:
        raise ValueError("The custom expression is too long (maximum 2000 characters).")
    tree = ast.parse(expression, mode="eval")
    nodes = list(ast.walk(tree))
    if len(nodes) > 500:
        raise ValueError("The expression is too complex.")
    for node in nodes:
        if not isinstance(
            node,
            (
                ast.Expression,
                ast.BinOp,
                ast.UnaryOp,
                ast.Call,
                ast.Name,
                ast.Load,
                ast.Constant,
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.Pow,
                ast.USub,
                ast.UAdd,
            ),
        ):
            raise ValueError("Expressions may only use X/Y/Z, pi, numbers, arithmetic operators, and the listed functions.")
        if isinstance(node, ast.Name) and node.id not in {*FUNCTIONS, "X", "Y", "Z", "pi"}:
            raise ValueError(f"Unsupported name: {node.id}")
        if isinstance(node, ast.Constant) and (
            type(node.value) not in (int, float) or abs(node.value) > 1e6
        ):
            raise ValueError("Expression constants must be numbers with absolute value no greater than 1e6.")
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name)
            or node.func.id not in FUNCTIONS
            or len(node.args) != 1
            or node.keywords
        ):
            raise ValueError("Functions must be single-argument mathematical functions.")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if (
                not isinstance(node.right, ast.Constant)
                or not isinstance(node.right.value, (int, float))
                or abs(node.right.value) > 8
            ):
                raise ValueError("Exponents must be numeric constants between -8 and 8.")
    return tree.body


def evaluate(expression, X, Y, Z):
    variables = {"X": X, "Y": Y, "Z": Z, "pi": np.pi}

    def visit(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError("A function name must be followed by parentheses")
            return variables[node.id]
        if isinstance(node, ast.BinOp):
            return OPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp):
            return -visit(node.operand) if isinstance(node.op, ast.USub) else visit(node.operand)
        if isinstance(node, ast.Call):
            return FUNCTIONS[node.func.id](visit(node.args[0]))
        raise ValueError("Unsupported expression node.")

    with np.errstate(all="ignore"):
        value = np.asarray(visit(parse_expression(expression)), dtype=np.float64)
    if not np.all(np.isfinite(value)):
        raise ValueError("The expression produced non-finite values in the sampling range. Check division, square roots, and exponents.")
    return value


@dataclass(frozen=True)
class Config:
    family: str = "Gyroid"
    mode: str = "sheet"
    size: tuple = (5.0, 5.0, 5.0)
    cells: tuple = (1, 1, 1)
    density_start: float = 0.25
    density_end: float = 0.50
    gradient: str = "linear"
    axis: str = "z"
    shape: str = "box"
    resolution: int = 16
    quality_strategy: str = "quality_fan"
    domain_mode: str = "solid"
    periodic_amplitudes: tuple = (0.05, 0.05, 0.05)
    phase_degrees: tuple = (0.0, 0.0, 90.0)
    custom: str = "sin(X)*cos(Y)+sin(Y)*cos(Z)+sin(Z)*cos(X)"

    def __post_init__(self):
        for name in ("size", "cells", "periodic_amplitudes", "phase_degrees"):
            value = getattr(self, name)
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"{name} must be a sequence of length 3")
            object.__setattr__(self, name, tuple(value))

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("Parameters must be a JSON object.")
        try:
            c = cls(**data)
        except TypeError as exc:
            raise ValueError("Unknown parameter fields were provided.") from exc
        c.validate()
        return c

    def validate(self):
        if self.domain_mode not in ("solid", "solid_fluid"):
            raise ValueError("domain_mode must be solid or solid_fluid")
        if self.quality_strategy not in ("pulling", "quality_fan"):
            raise ValueError("quality_strategy must be pulling or quality_fan")
        if self.family not in FAMILIES and self.family != "Custom":
            raise ValueError("Unknown surface family.")
        if self.mode not in ("sheet", "solid_above", "solid_below"):
            raise ValueError("Unknown structure mode.")
        if self.gradient not in (
            "uniform",
            "linear",
            "quadratic",
            "cubic",
            "cosine",
            "exponential",
            "radial",
            "periodic",
        ):
            raise ValueError("Unknown gradient type.")
        if self.axis not in ("x", "y", "z"):
            raise ValueError("Unknown gradient axis.")
        if self.shape != "box":
            raise ValueError("Unknown outer shape.")
        if len(self.size) != 3 or any(
            type(v) not in (int, float) or not np.isfinite(v) or not 0.1 <= v <= 1000
            for v in self.size
        ):
            raise ValueError("All three outer dimensions must be between 0.1 and 1000 mm.")
        if len(self.cells) != 3 or any(type(v) is not int or not 1 <= v <= 6 for v in self.cells):
            raise ValueError("Cell counts must be integers from 1 to 6.")
        if type(self.resolution) is not int or not 8 <= self.resolution <= 64:
            raise ValueError("Resolution per cell must be an integer from 8 to 64.")
        if np.prod(np.array(self.cells) * self.resolution) > 200_000:
            raise ValueError("The background grid exceeds 200,000 cubes. Reduce the cell counts or resolution.")
        for v in (self.density_start, self.density_end):
            if type(v) not in (int, float) or not np.isfinite(v) or not 0.08 <= v <= 0.85:
                raise ValueError("Relative density must be between 0.08 and 0.85.")
        if len(self.periodic_amplitudes) != 3 or any(
            type(v) not in (int, float) or not np.isfinite(v) or abs(v) > 0.4
            for v in self.periodic_amplitudes
        ):
            raise ValueError("Periodic gradient amplitudes must be three finite numbers with absolute value no greater than 0.4.")
        if self.gradient == "periodic" and (
            self.density_start - sum(map(abs, self.periodic_amplitudes)) < 0.08 - 1e-10
            or self.density_start + sum(map(abs, self.periodic_amplitudes)) > 0.85 + 1e-10
        ):
            raise ValueError("The periodic gradient density range exceeds 0.08 to 0.85. Reduce the amplitudes or adjust the mean density.")
        if len(self.phase_degrees) != 3 or any(
            type(v) not in (int, float) or not np.isfinite(v) or abs(v) > 360
            for v in self.phase_degrees
        ):
            raise ValueError("Phases must be three numbers between -360 and 360 degrees.")
        parse_expression(self.expression)

    @property
    def expression(self):
        return self.custom if self.family == "Custom" else FAMILIES[self.family]["expression"]

    def to_dict(self):
        return asdict(self)


@lru_cache(maxsize=64)
def calibration(expression, mode):
    # Midpoints avoid bias from duplicated periodic endpoints.
    a = (np.arange(56) + 0.5) * (2 * np.pi / 56)
    values = evaluate(expression, a[:, None, None], a[None, :, None], a[None, None, :])
    values = np.broadcast_to(values, (56, 56, 56))
    if mode == "sheet":
        values = np.abs(values)
    elif mode == "solid_above":
        values = -values
    values = np.sort(values.ravel())
    if np.ptp(values) < 1e-10:
        raise ValueError("The expression defines a constant field and cannot generate a structure at the requested density.")
    return np.linspace(0, 1, len(values)), values


def target_density(config, X, Y, Z):
    pos = (X, Y, Z)
    if config.gradient == "periodic":
        return config.density_start + sum(
            a * np.cos(2 * np.pi * p / L)
            for a, p, L in zip(config.periodic_amplitudes, pos, config.size)
        )
    k = "xyz".index(config.axis)
    t = np.clip(pos[k] / config.size[k], 0, 1)
    if config.gradient == "uniform":
        t = np.zeros_like(t)
    elif config.gradient == "quadratic":
        t = t**2
    elif config.gradient == "cubic":
        t = t**3
    elif config.gradient == "cosine":
        t = (1 - np.cos(np.pi * t)) / 2
    elif config.gradient == "exponential":
        t = np.expm1(3 * t) / np.expm1(3)
    elif config.gradient == "radial":
        axes = [i for i in range(3) if i != k]
        radius = min(config.size[i] for i in axes) / 2
        t = np.clip(np.sqrt(sum((pos[i] - config.size[i] / 2) ** 2 for i in axes)) / radius, 0, 1)
    return config.density_start + (config.density_end - config.density_start) * t


def raw_field(config, X, Y, Z):
    angles = [
        2 * np.pi * p * n / L + np.deg2rad(phase)
        for p, n, L, phase in zip((X, Y, Z), config.cells, config.size, config.phase_degrees)
    ]
    f = evaluate(config.expression, *angles)
    probabilities, values = calibration(config.expression, config.mode)
    q = np.interp(target_density(config, X, Y, Z), probabilities, values)
    if config.mode == "sheet":
        f = np.abs(f)
    elif config.mode == "solid_above":
        f = -f
    return q - f  # Positive means solid material.
