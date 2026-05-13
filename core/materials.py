"""Material database and Cross-WLF viscosity model."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Material:
    key: str
    name: str
    n: float
    tau_star: float  # Pa
    D1: float  # Pa.s
    D2: float  # K
    D3: float  # K/Pa
    A1: float  # -
    A2_tilde: float  # K
    T_melt_recommended: tuple[float, float]
    T_mold_recommended: tuple[float, float]
    density_melt_kgm3: float
    # Thermal diffusivity α [m^2/s], used by the skin-layer (Stefan/Neumann)
    # model: s(t) = c_skin · sqrt(α · t). Generic values for educational use;
    # 1e-7 m^2/s is a typical melt-polymer order of magnitude.
    thermal_diffusivity_m2_s: float = 1.0e-7
    # Specific heat capacity at the melt state [J/(kg·K)]. Used by the
    # shear-heating (viscous dissipation) correction in the multilayer
    # solver: ΔT/Δt = η·γ̇² / (ρ·cp). 2400 J/(kg·K) is a generic PP-melt
    # default; talc-filled grades have lower values (lower for higher
    # filler fraction). Together with ``density_melt_kgm3`` and
    # ``thermal_diffusivity_m2_s`` the thermal conductivity is recovered
    # as ``k = α · ρ · cp`` (used for the Brinkman number).
    specific_heat_J_kgK: float = 2400.0

    @property
    def thermal_conductivity_W_mK(self) -> float:
        """Derived thermal conductivity ``k = α · ρ · cp`` [W/(m·K)].

        Derived from existing fields so the material DB doesn't need a
        separate column. For PP this gives ``≈ 0.16 W/(m·K)`` which is
        in the right ballpark for molten polypropylene.
        """
        return float(
            self.thermal_diffusivity_m2_s * self.density_melt_kgm3 * self.specific_heat_J_kgK
        )


class MaterialDB:
    """Loads materials from JSON, exposes lookup helpers."""

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = Path(__file__).resolve().parent.parent / "data" / "materials.json"
        self._path = Path(path)
        with self._path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        self._meta = payload.get("_meta", {})
        raw = payload.get("materials", {})
        self._materials: dict[str, Material] = {}
        for key, m in raw.items():
            self._materials[key] = Material(
                key=key,
                name=m["name"],
                n=m["n"],
                tau_star=m["tau_star"],
                D1=m["D1"],
                D2=m["D2"],
                D3=m.get("D3", 0.0),
                A1=m["A1"],
                A2_tilde=m["A2_tilde"],
                T_melt_recommended=tuple(m["T_melt_recommended"]),
                T_mold_recommended=tuple(m["T_mold_recommended"]),
                density_melt_kgm3=m.get("density_melt_kgm3", 1000.0),
                thermal_diffusivity_m2_s=m.get("thermal_diffusivity_m2_s", 1.0e-7),
                specific_heat_J_kgK=m.get("specific_heat_J_kgK", 2400.0),
            )

    @property
    def meta(self) -> dict:
        return dict(self._meta)

    def keys(self) -> Iterable[str]:
        return self._materials.keys()

    def get(self, key: str) -> Material:
        return self._materials[key]

    def __getitem__(self, key: str) -> Material:
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self._materials

    def __iter__(self):
        return iter(self._materials.values())


def cross_wlf_viscosity(
    material: Material,
    temperature_K: float | np.ndarray,
    shear_rate: float | np.ndarray,
    pressure_Pa: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """Cross-WLF viscosity [Pa.s].

    eta(gamma_dot, T, P) = eta0(T,P) / (1 + (eta0*gamma_dot / tau*)^(1-n))
    eta0(T,P) = D1 * exp(-A1*(T-T*)/(A2_tilde + (T-T*)))
    T* = D2 + D3 * P
    """
    T = np.asarray(temperature_K, dtype=float)
    g = np.asarray(shear_rate, dtype=float)
    P = np.asarray(pressure_Pa, dtype=float)

    T_ref = material.D2 + material.D3 * P
    dT = T - T_ref
    # numerical guard: 分母が負やゼロにならないようクリップ
    denom = material.A2_tilde + dT
    denom = np.where(denom <= 1e-6, 1e-6, denom)
    eta0 = material.D1 * np.exp(-material.A1 * dT / denom)

    # avoid division by zero in shear rate
    g_safe = np.where(g <= 1e-12, 1e-12, g)
    ratio = (eta0 * g_safe) / material.tau_star
    eta = eta0 / (1.0 + ratio ** (1.0 - material.n))
    return eta


def representative_shear_rate(injection_velocity_mms: float, thickness_mm: float) -> float:
    """Representative wall shear rate for Hele-Shaw flow [1/s].

    gamma_dot ~ 6 * V_avg / h (Newtonian plate approximation as upper-bound proxy).
    """
    V = max(injection_velocity_mms * 1e-3, 1e-6)  # m/s
    h = max(thickness_mm * 1e-3, 1e-5)  # m
    return 6.0 * V / h
