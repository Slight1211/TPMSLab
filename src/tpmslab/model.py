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


@lru_cache(maxsize=128)
def _compiled(expression):
    """Bytecode for a whitelisted expression (see :func:`parse_expression`).

    Only X/Y/Z, pi, numeric constants, + - * / **, unary +/- and the listed
    single-argument numpy functions can appear, and names resolve only in the
    namespace passed to ``eval`` (no builtins), so this evaluates exactly the
    operations the former tree walker performed, in the same order.
    """
    body = parse_expression(expression)
    for node in ast.walk(body):
        if isinstance(node, ast.Name) and node.id in FUNCTIONS:
            parent_calls = [n for n in ast.walk(body) if isinstance(n, ast.Call) and n.func is node]
            if not parent_calls:
                raise ValueError("A function name must be followed by parentheses")
    return compile(ast.Expression(body=body), "<tpms expression>", "eval")


def evaluate(expression, X, Y, Z):
    namespace = {"__builtins__": {}, "X": X, "Y": Y, "Z": Z, "pi": np.pi, **FUNCTIONS}
    with np.errstate(all="ignore"):
        value = np.asarray(eval(_compiled(expression), namespace), dtype=np.float64)
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
    edge_interpolation: str = "linear"
    domain_mode: str = "solid"
    periodic_amplitudes: tuple = (0.05, 0.05, 0.05)
    phase_degrees: tuple = (0.0, 0.0, 90.0)
    custom: str = "sin(X)*cos(Y)+sin(Y)*cos(Z)+sin(Z)*cos(X)"

    m_cal: int = 56
    combine: str = "none"
    family_secondary: str = "Diamond"
    custom_secondary: str = "cos(X)+cos(Y)+cos(Z)"
    blend_axis: str = "z"
    blend_start: float = 0.25
    blend_end: float = 0.75
    blend_profile: str = "smoothstep"
    blend_calibration_steps: int = 17
    snap_alpha: float = 0.0
    snap_target: str = "implicit"
    lattice: str = "kuhn"
    snap_alpha_boundary: float | None = None
    periodic_faces: bool = False

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
        if self.edge_interpolation not in ("linear", "implicit_root", "quadratic_root"):
            raise ValueError("edge_interpolation must be linear, implicit_root, or quadratic_root")
        if self.edge_interpolation != "linear" and self.domain_mode != "solid_fluid":
            raise ValueError("root-based interpolation requires solid_fluid domain mode")
        if type(self.snap_alpha) not in (int, float) or not np.isfinite(self.snap_alpha) \
                or not 0 <= self.snap_alpha < 0.5:
            raise ValueError("snap_alpha must be a number in [0, 0.5).")
        if self.snap_target not in ("linear", "implicit"):
            raise ValueError("snap_target must be linear or implicit")
        if self.snap_alpha_boundary is not None and (
                type(self.snap_alpha_boundary) not in (int, float) or not np.isfinite(self.snap_alpha_boundary)
                or not 0 <= self.snap_alpha_boundary < 0.5):
            raise ValueError("snap_alpha_boundary must be None or a number in [0, 0.5).")
        if type(self.periodic_faces) is not bool:
            raise ValueError("periodic_faces must be true or false")
        if self.lattice not in ("kuhn", "bcc"):
            raise ValueError("lattice must be kuhn or bcc")
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
        if len(self.cells) != 3 or any(type(v) is not int or v < 1 for v in self.cells):
            raise ValueError("Cell counts must be positive integers.")
        if type(self.resolution) is not int or self.resolution < 8:
            raise ValueError("Resolution per cell must be an integer of at least 8.")
        if type(self.m_cal) is not int or self.m_cal < 8:
            raise ValueError("m_cal must be an integer of at least 8.")
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
        if self.combine not in ("none", "blend"):
            raise ValueError("combine must be none or blend")
        if self.combine == "blend":
            if self.family_secondary not in FAMILIES and self.family_secondary != "Custom":
                raise ValueError("Unknown secondary surface family.")
            if self.blend_axis not in ("x", "y", "z"):
                raise ValueError("blend_axis must be x, y, or z")
            if self.blend_profile not in ("smoothstep", "linear", "cosine"):
                raise ValueError("blend_profile must be smoothstep, linear, or cosine")
            if any(type(v) not in (int, float) or not np.isfinite(v)
                   for v in (self.blend_start, self.blend_end)):
                raise ValueError("Blend transition bounds must be finite numbers.")
            if not 0 <= self.blend_start < self.blend_end <= 1:
                raise ValueError("Blend transition requires 0 <= blend_start < blend_end <= 1.")
            if type(self.blend_calibration_steps) is not int or self.blend_calibration_steps < 2:
                raise ValueError("blend_calibration_steps must be an integer of at least 2.")
            parse_expression(self.expression_secondary)

    @property
    def expression(self):
        return self.custom if self.family == "Custom" else FAMILIES[self.family]["expression"]

    @property
    def expression_secondary(self):
        return (self.custom_secondary if self.family_secondary == "Custom"
                else FAMILIES[self.family_secondary]["expression"])

    def to_dict(self):
        return asdict(self)


@lru_cache(maxsize=64)
def calibration(expression, mode, m_cal=56):
    if type(m_cal) is not int or m_cal < 8:
        raise ValueError("m_cal must be an integer of at least 8.")
    # Midpoints avoid bias from duplicated periodic endpoints.
    a = (np.arange(m_cal) + 0.5) * (2 * np.pi / m_cal)
    values = evaluate(expression, a[:, None, None], a[None, :, None], a[None, None, :])
    values = np.broadcast_to(values, (m_cal, m_cal, m_cal))
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


def blend_weight(config, X, Y, Z):
    """Return the secondary-family weight at physical coordinates in millimetres."""
    k = "xyz".index(config.blend_axis)
    t = np.clip(((X, Y, Z)[k] / config.size[k] - config.blend_start)
                / (config.blend_end - config.blend_start), 0., 1.)
    if config.blend_profile == "smoothstep":
        return np.clip(t**3 * (10 - 15*t + 6*t**2), 0., 1.)  # C2 quintic transition.
    if config.blend_profile == "cosine":
        return (1 - np.cos(np.pi*t)) / 2
    return t


@lru_cache(maxsize=8)
def blended_calibration(expression_a, expression_b, mode, m_cal, steps):
    """Fixed-weight periodic quantiles; not a physical layer-density guarantee."""
    if expression_a == expression_b:
        p, v = calibration(expression_a, mode, m_cal)
        return p, np.broadcast_to(v, (steps, len(v)))
    a = (np.arange(m_cal) + .5) * (2*np.pi / m_cal)
    angles = (a[:, None, None], a[None, :, None], a[None, None, :])
    first = np.broadcast_to(evaluate(expression_a, *angles), (m_cal,)*3)
    second = np.broadcast_to(evaluate(expression_b, *angles), (m_cal,)*3)
    rows = []
    for w in np.linspace(0., 1., steps):
        if w == 0. or w == 1.:
            _, values = calibration(expression_a if w == 0. else expression_b, mode, m_cal)
        else:
            values = (1-w)*first + w*second
            if mode == "sheet":
                values = np.abs(values)
            elif mode == "solid_above":
                values = -values
            values = np.sort(values.ravel())
            if np.ptp(values) < 1e-10:
                raise ValueError(f"Combined field is constant at blend weight {w:g}; choose different families or expressions.")
        rows.append(values)
    return np.linspace(0., 1., m_cal**3), np.asarray(rows)


def field_and_threshold(config, X, Y, Z):
    """Evaluate the design field and calibrated threshold for all meshing paths."""
    angles = [
        2 * np.pi * p * n / L + np.deg2rad(phase)
        for p, n, L, phase in zip((X, Y, Z), config.cells, config.size, config.phase_degrees)
    ]
    f = evaluate(config.expression, *angles)
    density = target_density(config, X, Y, Z)
    if config.combine == "none":
        probabilities, values = calibration(config.expression, config.mode, config.m_cal)
        return f, np.interp(density, probabilities, values)
    w = blend_weight(config, X, Y, Z)
    if config.expression == config.expression_secondary:
        p, values = calibration(config.expression, config.mode, config.m_cal)
        return f, np.interp(density, p, values)
    f = (1-w)*f + w*evaluate(config.expression_secondary, *angles)
    p, rows = blended_calibration(config.expression, config.expression_secondary, config.mode,
                                  config.m_cal, config.blend_calibration_steps)
    w, density = np.broadcast_arrays(w, density)
    coordinate = w * (config.blend_calibration_steps - 1)
    lower = np.minimum(np.floor(coordinate).astype(int), config.blend_calibration_steps - 2)
    fraction = coordinate - lower
    q = np.empty(w.shape, dtype=float)
    for i in np.unique(lower):
        mask = lower == i
        qa = np.interp(density[mask], p, rows[i])
        qb = np.interp(density[mask], p, rows[i+1])
        q[mask] = (1-fraction[mask])*qa + fraction[mask]*qb
    return f, q


def raw_field(config, X, Y, Z):
    f, q = field_and_threshold(config, X, Y, Z)
    if config.mode == "sheet":
        f = np.abs(f)
    elif config.mode == "solid_above":
        f = -f
    return q - f  # Positive means solid material.
