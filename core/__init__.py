"""mold-flow-sim core package."""

from .geometry import Geometry, build_demo_geometry, geometry_from_image
from .materials import MaterialDB, cross_wlf_viscosity
from .solver import FlowResult, HeleShawSolver
from .visualizer import export_frames, render_fill_animation, render_pressure_map, render_weldlines

__all__ = [
    "MaterialDB",
    "cross_wlf_viscosity",
    "Geometry",
    "build_demo_geometry",
    "geometry_from_image",
    "HeleShawSolver",
    "FlowResult",
    "render_fill_animation",
    "render_pressure_map",
    "render_weldlines",
    "export_frames",
]
