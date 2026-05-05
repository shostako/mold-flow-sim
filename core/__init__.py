"""mold-flow-sim core package."""

from .geometry import (
    FilmGateConfig,
    Geometry,
    build_demo_geometry,
    build_film_gate_geometry,
    geometry_from_image,
)
from .materials import MaterialDB, cross_wlf_viscosity
from .solver import FlowResult, HeleShawSolver
from .visualizer import export_frames, render_fill_animation, render_pressure_map, render_weldlines

__all__ = [
    "MaterialDB",
    "cross_wlf_viscosity",
    "Geometry",
    "FilmGateConfig",
    "build_demo_geometry",
    "build_film_gate_geometry",
    "geometry_from_image",
    "HeleShawSolver",
    "FlowResult",
    "render_fill_animation",
    "render_pressure_map",
    "render_weldlines",
    "export_frames",
]
