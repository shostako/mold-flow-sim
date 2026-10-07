# mold-flow-sim

[日本語](README.md) | English

[![CI](https://github.com/shostako/mold-flow-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/shostako/mold-flow-sim/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A Python simulator for injection-molding flow analysis, heavily simplified with the Hele-Shaw approximation,
the Cross-WLF viscosity model and the pseudo-conduction method. It is meant for early studies of thin plates and their gates,
for teaching, and for proofs of concept.

> **This is not a replacement for Moldflow or Moldex3D.** In-plane 3D flow, packing, crystallization, shrinkage and warpage are not modeled.
> Temperature and solidification through the thickness are only approximated by the skin-layer and multilayer models. Use a commercial CAE package for real mold-design decisions.

The user interface and most of the documentation are in Japanese. Changelog: [CHANGELOG.en.md](CHANGELOG.en.md)

## Demo

Hosted on the free tier of Streamlit Community Cloud. Nothing to install.

<https://mold-flow-sim.streamlit.app>

> Because it is the free tier, the first visit after a quiet period takes 30 seconds to a minute to wake the app.
> The computation runs on Streamlit Cloud's shared CPU (1 vCPU / 1 GB RAM), so a fine mesh with the multilayer model can take tens of seconds.

## What it does

**Analysis**

- Solves the pseudo fill-time field τ of a thin cavity on a 2D structured grid in a single elliptic solve, −∇·(S∇τ)=1
- Three wall-cooling models (the UI starts with multilayer, N=7)
  - None: isothermal, one representative viscosity
  - Skin layer: a Stefan/Neumann frozen front `s(t) = c_skin·√(αt)` grows from the walls and only the core `h_core = h - 2s` flows. The wall keeps cooling after the flow front has passed (exposure clock). Reports freeze-off cells and the short shot beyond them
  - Multilayer: splits the thickness into N layers and couples the 1D Neumann temperature profile with per-layer Cross-WLF viscosity by fixed-point iteration. The short shot is judged from the center-layer temperature. Includes a viscous-heating correction (stage 1, a closed-form local approximation) and a Brinkman-number diagnostic
- An equivalent model of injection-compression molding (ICM): the compression stroke widens the flow path and shortens the fill time
- Two-phase short-shot model: predicts a short shot made on purpose with a limited shot volume, using the machine settings as they are. It solves an injection phase (filling the mold-open gap up to the shot volume) and a compression phase (closing the mold to push the melt pool forward, conserving volume) with two linear solves. The skin-layer and multilayer models can run in the injection phase
- Injection conditions are either a direct injection rate or computed from screw diameter and injection speed (the default, which also supports multi-stage injection and the V/P switchover position)

**Geometry**

- Film gate 1 to 15: parametric input for a thickness-adjusting film gate (a gate block with a land, main ramp, coring, well and outer-wall line). Each comes with defaults taken from an actual drawing or a design revision
  - Fan-shaped (symmetric): coring, edge channel, cut-down behind the ramp, thicker land at both ends, graded ramp angle, variable land length (coat-hanger type)
  - One-sided (double flow length), split feed (twin mini fans, L-shaped runner), T-shaped runner
  - Numbers such as "0807" in the on-screen names are the dates of the drawings they came from
- Direct gate: a plain plate with a direct gate
- Profile gate: a JSON spec taken from a drawing (multiple fans and a runner are supported)

**Results**

- Fill animation (a player you can step frame by frame) and isochrones
- Relative pressure map, weld and meld lines (judged by the angle at which two flows meet), air-trap candidates
- Layer maps for the skin-layer and multilayer models, history animation of the two-phase short shot
- 3D view (Plotly)
- A ZIP of all results (images, `metadata.json`, the input settings as `settings.json`, and a standalone `player.html`)
- Automatic A3 PDF drawing of the gate block (1:1 plan view, four sections, dimension lines and notes). Film gate 1 to 15 and Profile gate only; not available for Direct gate
- IGES (3D) export of the resin side of the gate block (also Film gate 1 to 15 and Profile gate only). The exported file is read back and checked against the solver's shape cell by cell, and it is offered for download only on a match. It needs a CAD kernel (OCP), so it is for local installs only and is not in the public demo

Materials: PP / PP_T10 / PP_T20 / PP_T30 / ABS / PC / PA66 / PMMA (generic values in `data/materials.json`).

## What it does not do (known limitations)

| Item | Status |
|------|--------|
| In-plane 3D flow, jetting, corner vortices | Not supported (a fundamental limit of Hele-Shaw methods; needs full 3D FVM/FEM) |
| Packing stage | Not supported. Only filling is solved |
| Crystallization, shrinkage, warpage, residual stress | Not supported |
| Viscous heating stage 2 (energy equation through the thickness) | Not supported. Stage 1 is a closed-form local approximation and drifts where Br ≫ 1 |
| Feedback from the flow field to shear rate | Not supported. γ̇ varies by layer and cell but is derived from a single representative injection speed |
| Real time in the compression phase | Not supported. The compression phase only gives the order of advance. Freezing during compression and overlapping injection and compression are not handled |
| Absolute pressure field, required clamp force | Not supported. Pressure is normalized only (gate = 1, flow front = 0) |
| Convection within a layer | Not supported. The 1D Neumann solution is pure diffusion (breaks down for very thick parts, h > 4 mm) |
| Direct STL / STEP import | Not supported. Parametric shapes and JSON specs only |
| Midplane mesh (unstructured grid) | Not supported. Structured grid only |

## Roadmap

Done:

1. Foundations: packaging, CI, tests, verification against a 1D analytic solution
2. Geometry input: Film gate 1 to 15, Direct gate, JSON specs from drawings (Profile gate)
3. Transient heat and frozen layer: skin-layer model, N-layer multilayer model, viscous heating stage 1
4. Two-phase ICM (partial): the two-phase short-shot model. The regular fill time and pressure map still use the equivalent model
5. Input from machine settings: screw diameter, injection speed, multi-stage injection, V/P switchover
6. Practical output: results ZIP with the recorded inputs, automatic gate-block drawing, IGES export

Remaining:

1. Feedback from the flow field to the velocity field: derive local velocity from the solved τ and feed it back into γ̇
2. Numerical groundwork: iterative solvers (CG / AMG), mesh-convergence tests
3. Input and output: VTK export, documented sources for the material database and more materials
4. ICM, continued: connect the two-phase model to the main line and give the compression phase real time from the clamp speed. Freezing during compression comes after that
5. Viscous heating stage 2: an implicit 1D FDM solve of the energy equation through the thickness

## Installation

```bash
git clone https://github.com/shostako/mold-flow-sim.git
cd mold-flow-sim
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e .                     # runtime only
pip install -e ".[dev]"              # development (includes ruff and pytest)
pip install -e ".[cad]"              # for IGES export (OCP, about 160 MB)
```

Requires Python 3.11 or later.

> `requirements.txt` / `runtime.txt` / `packages.txt` / `.streamlit/config.toml` are deployment settings for Streamlit Community Cloud
> and are not used for local development (`pyproject.toml` is the source of truth).

## Running

### Streamlit UI

```bash
streamlit run app.py
```

Opens `http://localhost:8501` in the browser. Set the geometry, material, injection conditions and wall cooling in the left sidebar, then press 「解析実行」 (Run analysis).

### CLI batch (parameter sweeps)

```bash
python run_demo.py
python run_demo.py --cases PP_baseline PP_dual_gate
```

Writes GIFs, PNGs and numbered frames to `outputs/<case>/`.

### Tests and lint

```bash
pytest tests/
ruff check . && ruff format --check .
```

## Physical model

### Hele-Shaw approximation and the pseudo-conduction method

The filling process is solved as one elliptic problem instead of by time stepping:

```
−∇·(S ∇τ) = 1    (in the cavity)
τ = 0            (gate, Dirichlet)
S∇τ·n = 0        (walls, Neumann)
S = h³ / (12·η_eff)
```

The sign matches the discretization in the code (diagonal `+Σcoeff` / off-diagonal `−coeff` / right-hand side `+1`).
In continuous form, `∇·(S∇τ) = −1`.

With this sign, the operator before constraints is symmetric positive semidefinite (each face conductance is shared by both neighbors).
The assembled `A`, however, is neither symmetric nor positive definite: the Dirichlet condition is applied to rows only,
so the gate rows collapse to identity rows while the neighboring interior rows keep the `−coeff` in the gate columns.
Moving to CG / AMG as on the roadmap needs the gate columns eliminated first. The elimination is exact rather than an approximation
(τ = 0 at the gate, so the terms moved to the right-hand side are zero), but `spsolve` does not require symmetry, so it has not been done.

- `τ` is a pseudo arrival-time field (a monotonic function of "distance" from the gate)
- Conversion to absolute time uses the cumulative volume: `fill_time(x,y) = T_fill · V(τ' ≤ τ(x,y)) / V_solved`.
  At a constant injection rate the front advances in proportion to volume, so the volume accumulated in τ order gives the arrival time.
  `V_solved` is the volume of the cells that can fill; short-shot cells cut off by freeze-off are excluded
- `T_fill = V_solved / Q` holds at a constant rate without compression (no wall cooling, or the skin layer's velocity-controlled clock).
  The ICM equivalent model shortens the fill time, and the skin layer's constant-pressure clock and the multilayer thermal coupling rescale it by the ratio of volume-weighted τ
- Face conductance is the harmonic mean of the two neighboring cells

### Cross-WLF viscosity model

```
η(γ̇, T, P) = η₀(T,P) / (1 + (η₀ γ̇ / τ*)^(1−n))
η₀(T,P) = D₁ · exp(−A₁ (T−T*) / (Ã₂ + (T−T*)))
T* = D₂ + D₃ P
```

Material parameters are in `data/materials.json` (generic values).

### Multilayer Hele-Shaw solver

The thickness is split into `N` layers, each with its own temperature, viscosity and shear rate. Layer temperatures come from a superposition of 1D Neumann solutions,
Cross-WLF turns them into layer viscosities, and the lubrication integral for a viscosity that varies across the thickness combines them into one conductance (each layer weighted by its squared distance from the midplane):

```
T(z, t) = T_mold + (T_melt - T_mold) · [erf(z/(2√(αt))) + erf((h-z)/(2√(αt))) - 1]
γ̇_k(x,y) = (6V/h) · |2ζ_k - 1|                               # analytic Poiseuille derivative
η_k(x,y) = cross_wlf_viscosity(material, T_k, γ̇_k, 0)
S_total(x,y) = ∫(z − h/2)²/η dz = h³ · Σ_k m_k / η_k          # m_k = [(ζ − 1/2)³/3], Σ m_k = 1/12
```

`τ ↔ T_k ↔ η_k ↔ S_total` are coupled by fixed-point iteration, and `T_fill` is scaled by the ratio of the volume-weighted mean τ (after iteration / baseline).
Cells whose center-layer temperature drops below the solidification threshold are marked as short shot.
Call it with `MultilayerHeleShawSolver(num_layers=5, layer_distribution="wall_refined", thermal_coupling=True)`.
With `num_layers=1` and `thermal_coupling=False` it is numerically identical to `HeleShawSolver` (covered by tests).

## License

MIT License. See [LICENSE](LICENSE).

The material parameters in `data/materials.json` are generic values for teaching. Use measured data from the material supplier for real work.
