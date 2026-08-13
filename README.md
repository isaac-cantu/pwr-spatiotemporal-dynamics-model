<div align="center">

# Reactor Core Latent World Model

**A multi-fidelity deep-learning surrogate for long-horizon thermal-hydraulic simulation of a PWR core**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![Report](https://img.shields.io/badge/report-PDF-red)](docs/report/report.pdf)

</div> 

---

A **world model** that learns to autoregressively predict nine stacked axial layers of a
15×15 thermal field for a four-loop pressurized-water reactor (PWR) core, trained under a
strict data-scarcity constraint — only **10%** of the available high-fidelity trajectory may
be used for gradient-based training. On a full real-data run, it reaches **sub-1% MAPE for 8
of 9 layers** at rollout horizons beyond 1,000 autoregressive steps, an estimated
**4–5 order-of-magnitude speed-up** over the underlying CFD simulation.

This repository is the current version, **v7.33**, together with the complete development
history and a 10-page technical report (literature review, methodology, real-data results,
and an honest account of open issues).

## Contents

- [Results at a glance](#results-at-a-glance)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Data setup](#data-setup)
- [Usage](#usage)
- [Model summary](#model-summary)
- [Known limitations](#known-limitations)
- [Documentation](#documentation)
- [License](#license)
- [Citation](#citation)

## Results at a glance

MAPE (%) by layer and rollout horizon, free autoregressive rollout, real high-fidelity test
split (full run log and diagnostic plots in [`docs/report/report.pdf`](docs/report/report.pdf)):

| Layer | t+10 | t+100 | t+500 | t+1000 |
|---|---|---|---|---|
| $L_0$ (turbulent inlet) | 0.79 | 4.14 | 5.21 | 5.29 |
| $L_1$ | 0.08 | 0.34 | 0.53 | 0.59 |
| $L_2$ | 0.02 | 0.13 | 0.23 | 0.19 |
| $L_3$–$L_7$ | ≤0.08 | ≤0.08 | ≤0.08 | ≤0.06 |
| $L_8$ | 0.02 | 0.06 | 0.15 | 0.16 |

$L_0$'s ~5% plateau is interpreted as a physical floor rather than a model deficiency (two
valid turbulent realizations of the same process decorrelate pointwise at long horizons even
when statistically indistinguishable); the report validates this with a dedicated
turbulence-statistics diagnostic rather than relying on pointwise MAPE alone.

## Repository structure

```
.
├── README.md                  <- you are here
├── CHANGELOG.md                <- full v5 -> v7.33 development history
├── LICENSE
├── requirements.txt
├── .gitignore
├── docs/
│   ├── report/
│   │   ├── report.tex          <- LaTeX source
│   │   └── report.pdf          <- compiled report (10 pages + references)
│   └── figures/                <- architecture/pipeline diagrams, result figures
├── data/                       <- NOT committed; see Data setup below
├── src/
│   ├── reactor_world_model.py               <- data loading, model, losses, training, execution
│   └── reactor_world_model_diagnostics.py   <- evaluation + diagnostic plots/video
└── outputs/                    <- NOT committed; checkpoints, plots, videos (git-ignored)
```

`data/` and `outputs/` are excluded from version control (`.gitignore`) since HDF5
trajectories, checkpoints, and generated plots are large binary artifacts that don't belong
in git history.

## Installation

Requires Python 3.9+ and a working PyTorch install (CUDA optional but recommended for
training; the code runs on CPU for the diagnostics/inference path).

```bash
git clone https://github.com/isaac-cantu/pwr-spatiotemporal-dynamics-model.git
cd pwr-spatiotemporal-dynamics-model
pip install -r requirements.txt
```

## Data setup

The code expects three folders of per-layer HDF5 files — one per fidelity level (low /
medium / high mesh resolution) — under `data/`, set by `FIDELITY_DIRS` near the top of
`reactor_world_model.py`:

```python
FIDELITY_DIRS = {
    "low":    "../data/hdf5_o10/",
    "medium": "../data/hdf5_o12/",
    "high":   "../data/hdf5/",
}
```

<details>
<summary><b>Expand for the exact folder/file layout and format details</b></summary>

```
data/
├── hdf5_o10/        <- low-fidelity (coarsest mesh)
│   ├── plane_01_Base.h5
│   ├── plane_02_Layer1.h5
│   ├── plane_03_Layer2.h5
│   ├── plane_04_Layer3.h5
│   ├── plane_05_Layer4.h5
│   ├── plane_06_Layer5.h5
│   ├── plane_07_Layer6.h5
│   ├── plane_08_Layer7.h5
│   └── plane_09_Layer8.h5
├── hdf5_o12/        <- medium-fidelity (same 9 filenames)
└── hdf5/            <- high-fidelity, ground truth (same 9 filenames)
```

- **Filenames must match exactly** (`plane_01_Base.h5` … `plane_09_Layer8.h5` — this list is
  `FILES` in `reactor_world_model.py`; edit it there if your naming differs).
- **Inside each `.h5`**: a single dataset of shape `(T, H, W)`. The loader auto-detects the
  dataset name if it's one of `data`, `value`, `values`, `field`, or the only dataset present
  — otherwise pass `dataset_key=` explicitly to `load_core_fidelity`. An optional 1-D `time`
  dataset lets the loader infer the real physical timestep; without it, `DT_PINN` defaults to
  `1.0`.
- Layers are resampled to `15×15` on load regardless of native resolution.
- Paths are relative to the working directory: run from `src/` (so `../data/` resolves
  correctly), or edit `FIDELITY_DIRS` to drop the `../` if running from the repo root.
- **No data yet?** If `data/hdf5/` isn't found, the script automatically falls back to a
  synthetic 3-fidelity dataset, so the full pipeline can be smoke-tested without real data.

</details>

## Usage

```bash
cd src
python3 reactor_world_model.py
```

This runs the full six-stage training curriculum and saves a checkpoint to
`outputs/checkpoint_v7_33.pt`. Then, as a **separate command** (no need to keep the same
session open):

```bash
python3 reactor_world_model_diagnostics.py
```

This automatically loads the saved checkpoint, re-evaluates the model, and writes every
figure and the rollout video to `outputs/plots_v6/`.

<details>
<summary>Notebook / interactive-session workflow</summary>

Both scripts were originally written as notebook cells and can still be pasted into the same
kernel session one after another — the diagnostics script detects an in-memory `model` and
skips the checkpoint reload automatically.

</details>

## Model summary

- **Encoder** — compact CNN → 64-d latent, conditioned on a fidelity embedding.
- **Deterministic dynamics** — causal Transformer over a 5-frame context, additionally
  conditioned on a Hankel-DMD trend forecast (needed because a 5-step context cannot resolve
  the system's ~530-step dominant oscillation).
- **Stochastic residual** — Rectified-Flow velocity network for calibrated uncertainty.
- **Decoder** — bounded-delta convolutional decoder (clamped update, not unbounded
  regression), preventing several classes of long-horizon divergence by construction.
- **Training curriculum** — 4 main phases (autoencoder → one-step dynamics → rollout
  consistency → optional physics fine-tuning), each split into a cheap-data pretrain
  sub-phase and a fine-tune sub-phase on the scarce 10% high-fidelity budget.
- **14 auxiliary loss terms**, each introduced for a specific diagnosed failure mode and
  unit-tested in isolation before deployment (full list and equations in the report).

Full architecture diagram, loss formulas, and training-curriculum figure in
[`docs/report/report.pdf`](docs/report/report.pdf) (Sections 4–5).

## Known limitations

Reported candidly, with full detail and evidence in the report's *Limitations and Open
Issues* section:

- **Unresolved boundary artifact.** A ring/frame-shaped visual artifact in mid-to-upper
  layers persists at long rollout horizons despite five targeted correction attempts (one a
  temporary regression, caught and corrected). Current hypothesis: an architectural
  bottleneck in the decoder's upsampling path, confirmed on real data in this run's
  fixed-bias diagnostic.
- **Rectified-Flow diversity collapse.** In this run, the stochastic residual's diversity
  ratio dropped to 0.029 (from ~1.5 in earlier synthetic tests) — point forecasts are
  unaffected, but calibrated uncertainty quantification is currently non-functional.
- $L_0$ will not meet a literal 1%-everywhere pointwise target at long horizons for physical
  (turbulence-decorrelation), not model-quality, reasons — see the turbulence-statistics
  diagnostic in the report for the appropriate evaluation standard for this layer.
- Trend/envelope statistics are frozen estimates from a fixed window; results derive from a
  single trajectory per fidelity at 15×15 resolution — generalization across operating
  conditions, core geometries, and resolutions is untested.

See `CHANGELOG.md` for the complete version-by-version history behind every design decision,
including two fully documented and corrected regressions.

## Documentation

- [`docs/report/report.pdf`](docs/report/report.pdf) — full technical report: literature
  review, data & methods, complete real-data results with diagnostic figures, development
  process case studies, limitations, and future work.
- [`CHANGELOG.md`](CHANGELOG.md) — the authoritative "why" behind every mechanism in the
  code, v5 through v7.33.


## Citation

If you use this code or report, please cite it as:

```bibtex
@techreport{reactor_world_model_2026,
  title  = {A Multi-Fidelity Latent World Model for Long-Horizon Surrogate Simulation of Reactor Core Thermal Fields},
  author = {Isaac Cantú},
  year   = {2026},
  note   = {v7.33},
  url    = {https://github.com/isaac-cantu/pwr-spatiotemporal-dynamics-model}
}
```