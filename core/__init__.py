"""mold-flow-sim core package."""

from .geometry import (
    DirectGateConfig,
    FilmGate2Config,
    FilmGateConfig,
    Geometry,
    build_demo_geometry,
    build_direct_gate_geometry,
    build_film_gate2_geometry,
    build_film_gate_geometry,
    geometry_from_image,
)
from .materials import MaterialDB, cross_wlf_viscosity
from .multilayer_solver import MultilayerFlowResult, MultilayerHeleShawSolver
from .solver import FlowResult, HeleShawSolver
from .visualizer import (
    export_frames,
    render_core_layer_map,
    render_fill_animation,
    render_pressure_map,
    render_skin_layer_map,
    render_weldlines,
)
from .visualizer_3d import (
    build_fine_geometry,
    fine_refine_factor,
    refine_for_display,
    render_3d_fill_time,
    render_3d_pressure,
    render_3d_thickness_map,
)

__all__ = [
    "MaterialDB",
    "cross_wlf_viscosity",
    "Geometry",
    "FilmGateConfig",
    "FilmGate2Config",
    "DirectGateConfig",
    "build_demo_geometry",
    "build_film_gate_geometry",
    "build_film_gate2_geometry",
    "build_direct_gate_geometry",
    "geometry_from_image",
    "HeleShawSolver",
    "MultilayerHeleShawSolver",
    "MultilayerFlowResult",
    "FlowResult",
    "render_fill_animation",
    "render_pressure_map",
    "render_weldlines",
    "render_skin_layer_map",
    "render_core_layer_map",
    "export_frames",
    "render_3d_thickness_map",
    "render_3d_fill_time",
    "render_3d_pressure",
    "refine_for_display",
    "build_fine_geometry",
    "fine_refine_factor",
]
