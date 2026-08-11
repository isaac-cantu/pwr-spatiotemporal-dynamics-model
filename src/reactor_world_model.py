"""
===============================================================================
WORLD MODEL v7.33 -- Diffusion/neighbor-consistency loss + first-100-steps
                      rollout video (urgent delivery)
===============================================================================
This is the final-delivery version of the reactor-core latent world model
(v5 -> v7.33). The full, version-by-version development history -- every
diagnosed failure, every fix, every piece of evidence behind a design
decision -- lives in CHANGELOG.md at the repository root. Comments in this
file are kept short and point back to specific CHANGELOG.md entries by
version tag (e.g. "v7.20 -- ...") instead of duplicating that narrative.

WHAT'S NEW IN v7.33 (see CHANGELOG.md "v7.33" for the full rationale):
  1. `diffusion_consistency_loss` (new, W_DIFFUSION=1.0) -- user's idea: a
     loss that checks how each grid point's evolution relates to its 4
     immediate in-plane neighbors, via the residual of the pure diffusion
     equation dC/dt = alpha*Laplacian(C). Does NOT demand a zero residual
     (unlike the historically ill-fitting Phase-3/PINN residual) -- it only
     penalizes the prediction's residual differing from reality's.
  2. First-100-steps rollout video (diagnostics script): Real | Predicted |
     MAPE(%), all 9 layers, one frame per step, none skipped.
  The persistent boundary ring artifact (see CHANGELOG.md v7.26-v7.32)
  remains an OPEN issue -- five statistics/envelope-side fixes have been
  tried; the leading hypothesis is a decoder architecture bottleneck
  (coarse bilinear upsampling), with a specific next experiment proposed in
  CHANGELOG.md and in the project report but not attempted here.
===============================================================================
"""

import os
import copy
import math
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 0. HYPERPARAMETERS
# ==============================================================================
SEQ_LEN            = 5          # context window (same as v5)
LATENT_DIM         = 64
BATCH_SIZE         = 16
GRAD_CLIP          = 1.0
MAX_DECODER_DELTA  = 0.15

# --- Phase 1: per-frame autoencoder ---------------------------------------
LR_AE               = 1e-3
LR_AE_FINETUNE      = 2e-4
EPOCHS_AE_PRETRAIN  = 30
# v7.15 -- raised 15->40: with only ~999 high-fidelity frames, more epochs
# over the SAME data is cheap (the bottleneck is data quantity, not compute
# per epoch). Protected by early stopping (`finetune_autoencoder`) against
# overfitting -- will normally stop well before 40 if Val MAPE plateaus.
EPOCHS_AE_FINETUNE  = 40
AE_FINETUNE_PATIENCE = 8   # v7.15 -- new: epochs without Val MAPE improvement before stopping Phase 1b
W_MSE   = 1.0
W_GRAD  = 5.0
# v7.15 -- new: see `spatial_spectral_loss`. Conservative weight for Phase 1
# (single-frame autoencoder, low/medium/high fidelity).
# v7.16 -- lowered 1.0->0.3 (see CHANGELOG.md v7.16): secondary hypothesis
# for why reconstruction worsened in the first real run -- the 2D FFT
# includes the mask's irregular boundary, which may compete with plain
# reconstruction more than expected. Raise carefully after validating.
W_SPATIAL_SPECTRAL_AE = 0.3
# v7.28 -- new: curvature-loss weight in Phase 1 (autoencoder). The block
# artifact was seen from t+10 (single-step rollout via Phase 1/2), so the
# fix must act already in basic reconstruction, not only in long rollouts.
# v7.29 -- raised 1.0->1.5: now the ONLY mechanism against blocking/mosaic
# (CoordConv was reverted, see CHANGELOG.md v7.29), so it's reinforced
# slightly. Unit-tested (v7.28): ~0 on identical fields, high on a field
# with injected blocks, finite gradients.
W_CURVATURE_AE = 1.5

# --- Phase 2: one-step dynamics (transformer + flow matching) ---------------
N_FLOW_STEPS        = 8
N_FLOW_STEPS_TRAIN  = 4
LR_DYN              = 2e-4
LR_DYN_FINETUNE     = 1e-4
EPOCHS_DYN_PRETRAIN = 30
EPOCHS_DYN_FINETUNE = 15

# --- Phase 2.5 (new): rollout training (exposure-bias fix) ---
RUN_PHASE_ROLLOUT      = True
# v7.27 -- 320 -> 512 (see CHANGELOG.md v7.27): the real v7.26 run showed L4
# orientation drifting from horizontal to DIAGONAL right after t+~500 --
# immediately past the max horizon the shape/orientation losses (anisotropy
# v7.25, centroid v7.15, correlation v7.22) can shape during training.
# Beyond K_ROLLOUT_STEPS the rollout drifts to the decoder's preferred
# attractor (the diagonal, a documented bias since v7.8/v7.17) with nothing
# correcting it. Extending the training horizon pushes that drift point
# further out. Cost: ~1.6x per sample, but the per-epoch batch cap (v7.19)
# keeps epoch time bounded; memory unchanged (TBPTT stays in 24-step chunks).
K_ROLLOUT_STEPS        = 512
K_ROLLOUT_STEPS_START  = 24
TBPTT_CHUNK             = 24
NOISE_STD_ROLLOUT       = 0.03
# v7.18 -- new: PHASE 2.5a -- rollout-CONSISTENCY pretraining (multi-step,
# not just one) on abundant low+medium fidelity, before fine-tuning on the
# 10% high-fidelity budget (Phase 2.5b, "Phase 2.5" in earlier versions).
# Same pretrain/finetune pattern already used by Phase 1 (1a/1b) and Phase 2
# (2a/2b) -- explicit user request: let the model learn each layer's physics
# in low fidelity first, so high-fidelity data only needs to refine it.
# Higher LR than fine-tuning (abundant data, larger steps are fine) and no
# early-stopping/checkpointing here (same as 1a/2a) -- careful checkpoint
# selection happens later, in 2.5b, on real high-fidelity data.
EPOCHS_ROLLOUT_PRETRAIN = 15
LR_ROLLOUT_PRETRAIN     = 2e-4
# v7.19 -- new: COST CONTROL for Phase 2.5a/2.5b (see CHANGELOG.md v7.19).
# With real data, low+medium fidelity have THOUSANDS of steps -> thousands
# of overlapping rollout windows -> each 2.5a epoch would traverse hundreds
# of batches, each with rollouts of up to 320 steps and backward -- a cost
# that can become impractical (hours/days per epoch) or exhaust memory. Two
# mechanisms, both applied to the rollout datasets:
#  1. `ROLLOUT_WINDOW_STRIDE`: step-by-step overlapping windows are
#     redundant (two windows 1 step apart share 99.7% of content at
#     k_steps=320) -- sampling every `stride` steps cuts the window count
#     ~stride-fold without losing real temporal coverage. Low/medium
#     fidelity only; high fidelity (scarce) keeps stride=1.
#  2. `MAX_ROLLOUT_BATCHES_PER_EPOCH`: hard per-epoch, per-fidelity batch
#     cap in 2.5a (random sampling via DataLoader shuffle) -- bounded epoch
#     time regardless of dataset size.
ROLLOUT_WINDOW_STRIDE          = 8
MAX_ROLLOUT_BATCHES_PER_EPOCH  = 30

# v7.19 -- new: EPOCH-LEVEL NUMERICAL SAFETY NET (see CHANGELOG.md v7.19).
# The v7.18 per-chunk guard discards NaN/Inf batches, but if an entire (or
# almost entire) epoch produces only discarded batches, training just spins
# in place. These parameters enable an automatic "rollback + LR backoff": a
# snapshot is saved at the start of each 2.5a/2.5b epoch; if the fraction of
# non-finite batches exceeds `NONFINITE_EPOCH_FRACTION`, the snapshot is
# RESTORED (discarding the corrupt epoch), LR is halved, and the epoch is
# retried. After `MAX_LR_BACKOFFS` consecutive reductions without a healthy
# epoch, the phase stops cleanly, keeping the last good state -- training
# can NEVER end with a corrupted model.
NONFINITE_EPOCH_FRACTION = 0.5
MAX_LR_BACKOFFS          = 3
# v7.15 -- raised 10->20, protected by early stopping + checkpoint (see
# `finetune_rollout_consistency`) -- same logic as EPOCHS_AE_FINETUNE.
# v7.21 -- raised 20->30: with training already stable (v7.18-v7.20: no
# NaNs, no trend bias) and per-epoch cost BOUNDED by
# MAX_ROLLOUT_BATCHES_PER_EPOCH (v7.19), more 2.5b epochs are cheap and
# attack the one remaining numeric target (L1 just above 1% at t+500+).
# Protected as always by early stopping + best long-horizon checkpoint (v7.17).
EPOCHS_ROLLOUT_FINETUNE = 30
ROLLOUT_PATIENCE        = 6   # v7.15 -- new: probe/val checks without improvement before stopping Phase 2.5
LR_ROLLOUT              = 1e-4
W_SHAPE  = 8.0
W_LEVEL  = 2.0
W_SPREAD = 4.0
W_SPECTRAL       = 4.0
W_VAR_FLOOR      = 10.0
VAR_FLOOR_RATIO  = 0.5
# v7.21 -- new: temporal-variance CEILING, symmetric counterpart to the
# floor (v6.1). Evidence: after the DMD fix (v7.20) predicted temporal std
# dropped from ~6x to ~1.5-2x real -- the floor prevents collapse to a
# constant, but nothing penalized EXCESS oscillation.
# v7.23 -- tightened (5->8, ratio 1.3->1.2): predicted oscillation amplitude
# still somewhat above real at short horizon.
W_VAR_CEIL       = 8.0
VAR_CEIL_RATIO   = 1.2
W_GROWTH         = 6.0
MAX_GROWTH_RATIO = 2.0

# v7.7/v7.8 -- see CHANGELOG.md: left at 0 by default (confirmed regression
# in a real run), code kept intact in case of careful re-testing.
# v7.23 -- REACTIVATED with a SMALL weight (0.0 -> 0.3; in v7.7 it was 1.0
# and caused a regression). Evidence from the real v7.22 run: bright ring at
# the mask PERIMETER in predicted L4-L5 at long horizon, with slight
# interior fragmentation -- spurious edge gradients that neither the
# per-pixel envelope (boundary pixels have legitimately wide real ranges)
# nor pattern correlation (a small fraction of pixels) penalize strongly.
# The gradient loss penalizes EXACTLY that: spatial-gradient differences
# pred vs. real. Context differs from v7.7: now `spatial_pattern_correlation`
# anchors global shape (it didn't exist in v7.7, and the gradient term
# competed against MSE unchecked), and checkpoint/early-stop/probe/rollback
# protect against any regression.
# v7.24 -- 0.3 -> 0.5: in the real v7.23 run the boundary ring shrank (the
# reactivation worked without a MAPE regression), but the interior of L4-L6
# at long horizon became "speckled" -- more gradient weight pushes toward
# real smoothness.
W_GRAD_ROLLOUT       = 0.5
W_SPATIAL_VAR        = 0.0
SPATIAL_VAR_MAX_RATIO = 1.5

# v7.15 -- new: weights for the two new Phase-2.5 losses. Conservative by
# design (see CHANGELOG.md v7.15) -- validate with
# `check_bifurcation_is_static`/`plot_centroid_tracking`/
# `plot_texture_directionality_check` before raising them.
# v7.16 -- W_SPATIAL_SPECTRAL lowered 2.0->1.0 as a precaution (same v7.7
# lesson: start low). W_CENTROID left UNCHANGED -- this piece DID work (see
# CHANGELOG.md): L1 centroid drift stayed under 0.85px over 1000 steps, vs.
# 1.5-2px divergent trajectories before.
# v7.17 -- lowered again, 1.0 -> 0.7: the real v7.16 run showed a NEW
# symptom -- several layers (L0-L2) now OVER-shoot real texture at long
# horizon (ratio >1.4-1.9). See also the new warm-up below
# (`SPATIAL_LOSS_WARMUP_EPOCHS`), attacking the same evidence from another
# angle.
# v7.22 -- DISABLED by default (0.7 -> 0.0), code intact (same pattern as
# W_GRAD_ROLLOUT/W_SPATIAL_VAR in v7.8). ROOT CAUSE IDENTIFIED in the real
# v7.21 run: 2D FFT magnitude is TRANSLATION-INVARIANT -- the model can put
# correct spectral energy in a SHIFTED band (observed: elongated band
# shifted left in L2-L5, where reality is a centered bulge) and this loss
# stays satisfied. The old reference model (v7.10, uploaded by the user) did
# NOT have this loss and its t+15 fields were visually faithful -- direct
# evidence that this loss, not something else, pushed the pattern out of
# place. Replaced by `spatial_pattern_correlation_loss` (below), which is
# position-sensitive.
W_SPATIAL_SPECTRAL = 0.0
# v7.22 -- new: spatial pattern correlation (pixel-wise Pearson between
# predicted vs. real DEVIATION fields, inside the mask). Unlike the spectral
# term (FFT magnitude), a SHIFTED pattern gives low correlation -- exactly
# "the same pattern in the same place." Spatial analog of
# `temporal_shape_loss` (W_SHAPE, v6), the temporal loss that has worked
# best throughout the project.
W_SPATIAL_CORR     = 4.0
W_CENTROID         = 1.0   # v7.18: lowered from 1.5 (extra caution -- the
                             # real fix is CENTROID_LAYER_WEIGHT, not this weight)

# v7.17 -- new: warm-up ramp for the two new v7.15 spatial losses
# (`W_SPATIAL_SPECTRAL`, `W_CENTROID`) in Phase 2.5 -- same pattern already
# used by `PINN_WARMUP_EPOCHS` for Phase 3. Motivation: in the real v7.16
# run, "quick Val MAPE" (short horizon) worsened MONOTONICALLY throughout
# Phase 2.5 -- applying these two losses at full strength from epoch 1
# (while the k_steps curriculum is still very short) may compete with the
# model first getting the basics (MSE/shape) right before refining
# texture/centroid.
SPATIAL_LOSS_WARMUP_EPOCHS = 8

# v7.17 -- new: horizon used to pick the BEST Phase-2.5 checkpoint
# (previously 15 steps, see `quick_rollout_val_mape`). Motivation (real
# evidence, v7.15/v7.16): a 15-step horizon is BLIND to the long-horizon
# improvement Phase 2.5 exists to give -- that made early stopping always
# prefer the EARLIEST epoch (least long-horizon training) because it looks
# best at 15 steps, not because it's actually best. Raised to 200 steps so
# the selection metric reflects the phase's real objective.
CHECKPOINT_EVAL_HORIZON = 200
# v7.21 -- raised 0.3 -> 0.6: the real v7.20 run showed the hot spot's
# LOCATION is already tracked well, but the prediction is TOO CONCENTRATED
# (tight/diagonal blob where reality is a wide, diffuse band, L3-L5) -- a
# SPREAD mismatch, not a position one. Raising this sub-weight pushes toward
# matching the real effective radius, without touching global W_CENTROID or
# the per-layer weight.
# v7.22 -- reverted 0.6 -> 0.3 (undoes the v7.21 raise): with the spectral
# loss pushing the pattern out of place, raising spread contributed to band
# ELONGATION. With the new correlation loss (which fixes position AND shape
# together), the original sub-weight is enough.
# v7.30 -- raised again 0.3 -> 0.5 (see CHANGELOG.md v7.30): real evidence
# shows the wide band keeps shrinking to a compact blob at long horizon --
# size (not just position/orientation) isn't being matched enough. This
# time the v7.21 risk (worsened elongation) is mitigated because a
# dedicated orientation/elongation loss now exists (`spatial_anisotropy_loss`,
# v7.25) that v7.21 didn't have.
CENTROID_SPREAD_SUBWEIGHT = 0.5

# v7.24 -- new (user's idea): INTER-LAYER coupling ("how lower layers affect
# upper ones"). Measures the Pearson correlation between the DEVIATION
# fields of each pair of adjacent layers and penalizes the predicted
# correlation matrix differing from the real one. The trainable counterpart
# to the axial coupling the PINN encodes as a hard equation
# (v_z*(T_i - T_{i-1})): instead of imposing an equation, it requires the
# observed RELATIONSHIP between layers to match the real data. Conservative
# weight; inside the same `SPATIAL_LOSS_WARMUP_EPOCHS` ramp as the other
# spatial losses.
W_INTERLAYER = 1.0

# v7.25 -- new (direct evidence from the flow field, see CHANGELOG.md
# v7.25): `centroid_spread_consistency_loss` only compares an ISOTROPIC
# radius (a circle) -- but L4 is a BAND, not a circle, and at long horizon
# the prediction loses horizontal orientation and becomes more
# radial/circular. `spatial_anisotropy_loss` extends hot-spot tracking from
# "centroid + radius" to "centroid + covariance ellipse" (major radius,
# minor radius, orientation) -- compares shape AND orientation, real vs.
# predicted. Conservative weight, same warm-up ramp and per-layer confidence
# weight as the centroid loss.
# v7.27 -- 1.0 -> 2.0: the real v7.26 run confirmed the anisotropy loss DID
# achieve elongation (the blob is no longer circular) but ORIENTATION is
# still wrong (diagonal where reality is horizontal) -- the failing axis is
# exactly what this loss penalizes, so its weight is raised (v7.25's unit
# tests showed healthy gradients and monotone optimization).
W_ANISOTROPY = 2.0

# v7.28 -- new: curvature-loss weight in Phase 2.5 (rollout). See
# CHANGELOG.md v7.28 and `spatial_curvature_loss` -- directly attacks the
# "block"/mosaic artifact reported in a real run. Inside the same warm-up
# ramp as the other spatial losses.
W_CURVATURE = 2.0   # v7.29: raised from 1.5, same reason as W_CURVATURE_AE

# v7.33 -- new (user's idea, see `diffusion_consistency_loss`): conservative
# weight, inside the same warm-up ramp as the other spatial losses. Reuses
# the already-calibrated `model.physics.alpha` -- does not demand a zero
# residual (Phase 3/PINN historically doesn't fit well), only that the
# prediction's residual resemble reality's.
W_DIFFUSION = 1.0

PROBE_HORIZON       = 400
PROBE_EVERY_EPOCHS  = 2
PROBE_DIVERGENCE_MAPE = 50.0

# --- Phase 3 (optional): physics fine-tuning (PINN) ------------------------
# v7.22 -- REACTIVATED (at the user's request) with safeguards the
# project's history demands (v7.4: "the residual never fit well and
# degraded rollouts"): W_PINN lowered 0.5->0.2, few epochs (6), full
# warm-up, physics CALIBRATED first on ground truth and frozen, NaN guard
# in the training step, and AUTOMATIC REVERSION: a model snapshot is saved
# before Phase 3, and if the subsequent stability probe diverges (or quick
# Val MAPE worsens by more than a relative PINN_ACCEPT_TOLERANCE), the
# snapshot is restored -- Phase 3 can no longer leave the model worse than
# it found it.
RUN_PHASE3_PINN      = True
EPOCHS_PINN_FINETUNE = 6    # v7.22: lowered from 10 (short, guarded phase)
LR_PINN_FINETUNE     = 1e-4
W_PINN               = 0.2   # v7.22: lowered from 0.5 (v6.1 lesson: a high weight dominated the gradient)
PINN_ACCEPT_TOLERANCE = 0.10  # v7.22: max. relative Val MAPE worsening tolerated after Phase 3
PINN_WARMUP_EPOCHS   = EPOCHS_PINN_FINETUNE
RUN_PHYSICS_CALIBRATION       = True   # v7.22: reactivated (consumer: Phase 3 reactivated)
EPOCHS_PHYSICS_CALIBRATION    = 5
LR_PHYSICS_CALIBRATION        = 1e-2
FREEZE_PHYSICS_AFTER_CALIBRATION = True

# --- Multi-fidelity -------------------------------------------------------
N_FIDELITIES = 3
FIDELITY_DIM = 16

# v7.15 -- new: data-augmentation switches. Applied to the SEQUENTIAL
# training loaders (Phase 2/2.5, all three fidelities) -- never to
# val/test/probe.
# v7.16 -- default changed to False (see CHANGELOG.md v7.16): in Phase 1
# (single-frame autoencoder, `FrameDatasetMF`) symmetry is now NEVER applied
# regardless of this flag (the call was removed from that class) because it
# confused the decoder (no pose signal -> it averaged the 8 orientations --
# reconstruction Val MAPE went from ~10% to ~17%, plus a cross-shaped visual
# artifact). This flag only controls the SEQUENTIAL loaders
# (`NextFrameDatasetMF`/`RolloutDatasetMF`), where the task stays
# well-defined under symmetry -- but it's left False by default there too
# until it's confirmed the (no-longer-confused) encoder doesn't carry the
# same problem. The 8 symmetries ARE valid for the real mask (confirmed in
# a real run) -- the mechanism is kept intact, just switched off.
USE_SYMMETRY_AUGMENTATION = False
SYMMETRY_AUGMENT_PROB     = 0.5     # probability of applying a non-identity symmetry, per sample (if re-enabled)
USE_NOISE_JITTER          = True
NOISE_JITTER_STD          = 0.005   # small relative to the real typical spatial std (~0.05-0.18, see TEXTURE_STD_TARGET)
NOISE_JITTER_PROB         = 0.5

# v7.15 -- new: centroid/spread model (parameters "found" in the data, same
# philosophy as the trend DMD).
CENTROID_EMBED_DIM     = 300
CENTROID_RESEED_EVERY  = 25

FILES = [
    "plane_01_Base", "plane_02_Layer1", "plane_03_Layer2", "plane_04_Layer3",
    "plane_05_Layer4", "plane_06_Layer5", "plane_07_Layer6", "plane_08_Layer7",
    "plane_09_Layer8",
]
FIDELITY_DIRS = {
    "low":  "../data/hdf5_o10/",
    "medium": "../data/hdf5_o12/",
    "high":  "../data/hdf5/",
}
FIDELITY_ID = {"low": 0, "medium": 1, "high": 2}
SAVE_DIR = "outputs/plots_v6"


# ==============================================================================
# 1. MULTI-FIDELITY DATA LOADING (no behavioral changes since v6)
# ==============================================================================
def _read_dataset(path, dataset_key=None, preferred_keys=("data", "value", "values", "field")):
    with h5py.File(path, "r") as f:
        if dataset_key is not None:
            return f[dataset_key][:]
        datasets = []
        f.visititems(lambda name, obj: datasets.append(name) if isinstance(obj, h5py.Dataset) else None)
        for key in preferred_keys:
            if key in datasets:
                return f[key][:]
        if len(datasets) == 1:
            return f[datasets[0]][:]
        elif len(datasets) == 0:
            raise ValueError(f"No dataset found in {path}")
        else:
            raise ValueError(f"{path} has multiple datasets {datasets}; specify dataset_key.")


def _resample_spatial(arr, target_hw):
    T, H, W = arr.shape
    target_h, target_w = target_hw
    if (H, W) == (target_h, target_w):
        return arr
    t = torch.tensor(arr, dtype=torch.float32).unsqueeze(1)
    t_resized = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return t_resized.squeeze(1).numpy()


def _read_time_axis(path, time_key="time"):
    with h5py.File(path, "r") as f:
        if time_key in f:
            return f[time_key][:]
        return None


def load_core_fidelity(hdf5_dir, files=FILES, target_hw=(15, 15), dataset_key=None, time_key="time"):
    layers = []
    dt_real = None
    for i, fname in enumerate(files):
        path = os.path.join(hdf5_dir, f"{fname}.h5")
        arr = _read_dataset(path, dataset_key=dataset_key).astype(np.float32)
        arr = _resample_spatial(arr, target_hw)
        layers.append(arr)
        if dt_real is None:
            t_axis = _read_time_axis(path, time_key=time_key)
            if t_axis is not None and len(t_axis) > 1:
                dt_real = float(np.mean(np.diff(t_axis)))
    T_lens = [l.shape[0] for l in layers]
    if len(set(T_lens)) > 1:
        T_min = min(T_lens)
        layers = [l[:T_min] for l in layers]
    return np.stack(layers, axis=1), dt_real


def normalize_reactor(arr, v_min, v_max):
    out = np.copy(arr)
    mask = out != 0.0
    shape = [1] * arr.ndim
    shape[-3] = 9
    v_min_b = np.asarray(v_min, dtype=np.float32).reshape(shape)
    v_max_b = np.asarray(v_max, dtype=np.float32).reshape(shape)
    normalized = (arr - v_min_b) / (v_max_b - v_min_b)
    return np.where(mask, normalized, 0.0).astype(np.float32)


def denormalize_tensor(t, v_min, v_max):
    out = t.clone()
    mask = out != 0.0
    shape = [1] * t.dim()
    shape[-3] = 9
    v_min_b = torch.as_tensor(v_min, dtype=t.dtype, device=t.device).reshape(shape)
    v_max_b = torch.as_tensor(v_max, dtype=t.dtype, device=t.device).reshape(shape)
    denorm = out * (v_max_b - v_min_b) + v_min_b
    return torch.where(mask, denorm, out)


TARGET_HW = (15, 15)
DT_PINN = 1.0

if os.path.isdir(FIDELITY_DIRS["high"]):
    print("Loading real multi-fidelity data from HDF5...")
    data_low, dt_low = load_core_fidelity(FIDELITY_DIRS["low"], target_hw=TARGET_HW)
    data_medium, dt_medium = load_core_fidelity(FIDELITY_DIRS["medium"], target_hw=TARGET_HW)
    data_high, dt_high = load_core_fidelity(FIDELITY_DIRS["high"], target_hw=TARGET_HW)
    if dt_high is not None:
        DT_PINN = dt_high
else:
    print("[!] No real HDF5 folders found -> generating 3 synthetic test fidelities.")

    def _make_synthetic(T, noise_scale, period=180.0):
        t_axis = np.arange(T)
        base = 80.0 + 4.0 * np.sin(2 * np.pi * t_axis / period) + 1.2 * np.sin(2 * np.pi * t_axis / (period / 2.3))
        signal_1d = base + np.random.randn(T) * 1.5
        arr = np.tile(signal_1d.reshape(T, 1, 1, 1), (1, 9, 15, 15)).astype(np.float32)
        layer_amp_scale = np.array([1.0, 0.35, 0.12, 0.05, 0.025, 0.015, 0.010, 0.008, 0.006], dtype=np.float32)
        arr = 80.0 + (arr - 80.0) * layer_amp_scale.reshape(1, 9, 1, 1)
        arr += (np.random.randn(*arr.shape).astype(np.float32) * noise_scale) * layer_amp_scale.reshape(1, 9, 1, 1)
        return arr

    data_high = _make_synthetic(1400, 0.3, period=180.0)
    data_medium = _make_synthetic(1200, 0.6, period=180.0)
    data_low = _make_synthetic(2000, 1.0, period=180.0)

mask_high = data_high != 0.0
v_min = np.full(9, np.inf, dtype=np.float32)
v_max = np.full(9, -np.inf, dtype=np.float32)
for _l in range(9):
    _vals = data_high[:, _l][mask_high[:, _l]]
    if _vals.size > 0:
        v_min[_l] = _vals.min()
        v_max[_l] = _vals.max()
v_max = np.maximum(v_max, v_min + 1e-6)
print("v_min per layer:", np.round(v_min, 3))
print("v_max per layer:", np.round(v_max, 3))

data_high_n = normalize_reactor(data_high, v_min, v_max)
data_medium_n = normalize_reactor(data_medium, v_min, v_max)
data_low_n = normalize_reactor(data_low, v_min, v_max)

# v7.31 -- new: reintroduces a STATISTICS window larger than the strict 10%
# used for gradient training (see CHANGELOG.md v7.31). Rationale: compared
# against an earlier version of this same project (v7.14, before the
# strict v7.17 split), which used 50% of the data for the DMD/envelope/
# centroid (closed-form, NO gradient -- SVD and percentiles, no overfitting
# risk from using more data there) and only 10% for gradient fine-tuning --
# and produced visually smoother dynamics. The persistent boundary band
# (CHANGELOG.md v7.30) survived THREE architecture/padding-side fix
# attempts, consistent with the real cause being STATISTICAL: with only
# ~10% of the data, per-pixel percentiles at boundary cells (the
# worst-sampled in the domain) are poorly conditioned regardless of decoder
# architecture.
#
# `HIGH_FIDELITY_STATS_FRAC` (new) separates again "how many steps are used
# to fit gradient-free statistics" from "how many steps the network sees
# during gradient fine-tuning." The latter (`HIGH_FIDELITY_TRAIN_FRAC`)
# stays at 10% -- the user's EXPLICIT training-budget constraint is
# unchanged. The former is raised to 25% (2.5x more data for the
# DMD/envelope/centroid) at ZERO training-budget cost. Train remains the
# LAST 10% of the statistics window (closest in time to val/test) -- same
# design v7.14 originally had.
HIGH_FIDELITY_STATS_FRAC = 0.25   # see CHANGELOG.md v7.31 -- ONLY for DMD/envelope/centroid, no training-budget cost
HIGH_FIDELITY_TRAIN_FRAC = 0.10   # gradient-based fine-tuning of the network -- strict, unchanged since v7.17
HIGH_FIDELITY_VAL_FRAC   = 0.10

N_high = len(data_high_n)
_min_train_len = SEQ_LEN + K_ROLLOUT_STEPS + 2
stats_end = max(_min_train_len, int(HIGH_FIDELITY_STATS_FRAC * N_high))
val_end = max(stats_end + 1, int((HIGH_FIDELITY_STATS_FRAC + HIGH_FIDELITY_VAL_FRAC) * N_high))
train_start = max(0, stats_end - int(HIGH_FIDELITY_TRAIN_FRAC * N_high))
train_start = min(train_start, stats_end - _min_train_len)

data_train_stats = data_high_n[:stats_end]
data_train = data_high_n[train_start:stats_end]
data_val = data_high_n[stats_end:val_end]
data_test = data_high_n[val_end:]
print(f"High fidelity -> stats (DMD/mask/envelope/centroid, NO gradient) = "
      f"{len(data_train_stats)} steps ({HIGH_FIDELITY_STATS_FRAC*100:.0f}% of N_high) | "
      f"train (GRADIENT fine-tuning, last {HIGH_FIDELITY_TRAIN_FRAC*100:.0f}% of N_high) = "
      f"{len(data_train)} steps | "
      f"val = {len(data_val)} steps ({HIGH_FIDELITY_VAL_FRAC*100:.0f}%) | "
      f"test = {len(data_test)} steps (~{100 - (HIGH_FIDELITY_STATS_FRAC + HIGH_FIDELITY_VAL_FRAC) * 100:.0f}%)")
if len(data_train) < SEQ_LEN + K_ROLLOUT_STEPS:
    print(f"[!] data_train ({len(data_train)} steps) is smaller than SEQ_LEN+K_ROLLOUT_STEPS "
          f"({SEQ_LEN + K_ROLLOUT_STEPS}) -- Phase 2.5 (RolloutDatasetMF) may end up EMPTY. "
          f"Raise HIGH_FIDELITY_TRAIN_FRAC/STATS_FRAC or lower K_ROLLOUT_STEPS.")


# ==============================================================================
# 1b. TREND CONDITIONING (DMD) -- v7
# ==============================================================================
def masked_layer_average_np(arr):
    if torch.is_tensor(arr):
        arr = arr.numpy()
    mask = arr != 0.0
    summed = (arr * mask).sum(axis=(2, 3))
    counts = np.clip(mask.sum(axis=(2, 3)), 1, None)
    return (summed / counts).astype(np.float32)


def _build_hankel_np(series, embed_dim):
    T0, n = series.shape
    n_cols = T0 - embed_dim + 1
    H = np.zeros((embed_dim * n, n_cols), dtype=np.float64)
    for i in range(n_cols):
        H[:, i] = series[i:i + embed_dim].reshape(-1)
    return H


def fit_dmd_trend(train_avg_series, embed_dim, energy_threshold=0.999, max_rank=150):
    """Exact DMD / HAVOK-style (Tu et al. 2014; Brunton et al.), generic over
    any (T, n) series -- also reused by `fit_centroid_dynamics` (v7.15) for
    centroid/spread, not only for the level trend."""
    layer_means = train_avg_series.mean(axis=0)
    centered = train_avg_series.astype(np.float64) - layer_means
    H = _build_hankel_np(centered, embed_dim)
    X, Xp = H[:, :-1], H[:, 1:]
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    cumulative_energy = np.cumsum(S ** 2) / np.sum(S ** 2)
    rank = int(np.searchsorted(cumulative_energy, energy_threshold) + 1)
    rank = max(4, min(rank, max_rank, len(S)))
    Ur, Sr, Vr = U[:, :rank], S[:rank], Vt[:rank, :].T
    A_tilde = Ur.T @ Xp @ Vr @ np.diag(1.0 / Sr)

    eigvals, eigvecs = np.linalg.eig(A_tilde)
    mags = np.abs(eigvals)
    scale = np.minimum(mags, 1.0) / np.maximum(mags, 1e-12)
    A_tilde_stable = np.real(eigvecs @ np.diag(eigvals * scale) @ np.linalg.inv(eigvecs))
    return A_tilde_stable, Ur, rank, layer_means.astype(np.float32)


def simulate_dmd_trend(A_tilde, Ur, seed_series, embed_dim, horizon, layer_means):
    n = seed_series.shape[1]
    centered_seed = seed_series[-embed_dim:].astype(np.float64) - layer_means
    z = Ur.T @ centered_seed.reshape(-1)
    preds = np.zeros((horizon, n), dtype=np.float32)
    for t in range(horizon):
        z = A_tilde @ z
        v_full = Ur @ z
        preds[t] = (v_full[-n:] + layer_means).astype(np.float32)
    return preds


def simulate_dmd_trend_rolling(A_tilde, Ur, avg_series_full, embed_dim, start_idx, total_horizon,
                                layer_means, reseed_every=25, local_recenter=True, crossfade_steps=5):
    """v7.20 -- TWO fixes based on real-run evidence (see CHANGELOG.md
    v7.20):

    1. `local_recenter=True` (new, default): instead of centering each
       segment on the GLOBAL average of the training window (`layer_means`
       -- which, with the 10/10/80 split, is only the first 10% of the
       series and is STALE relative to the real val/test level), it centers
       on the LOCAL average of that segment's seed. Bug mechanism this
       fixes: the DMD eigenvalues are stabilized to |lambda|<=1 (v7), so
       within each segment the forecast DECAYS toward its fixed point -- if
       that fixed point is the stale training average, each segment drags
       the level down, and since `apply_trend_renorm` pins the prediction to
       this trend, the bias is injected directly into the final field
       (observed: prediction systematically below reality on all 9 layers
       in the continuous simulation). The linear dynamics A are invariant to
       center translations, so using the local average is mathematically
       consistent and removes the stale fixed point.
    2. `crossfade_steps=5` (new): linear blend between the previous
       segment's last value and the new segment's first steps -- removes
       the "sawtooth" at reseed boundaries. Evidence: peaks of
       |d(trend)/dt| in `plot_transition_trigger_analysis` occur every ~25
       steps == `DMD_RESEED_EVERY`; via `apply_trend_renorm`, that sawtooth
       became high-frequency jitter in the prediction (predicted temporal
       std ~6x real in `plot_variance_collapse_check`)."""
    n = avg_series_full.shape[1]
    preds = np.zeros((total_horizon, n), dtype=np.float32)
    step = 0
    prev_last = None
    while step < total_horizon:
        seed_end = start_idx + step
        seed_start = max(0, seed_end - embed_dim)
        seed = avg_series_full[seed_start:seed_end]
        if len(seed) < embed_dim:
            pad_len = embed_dim - len(seed)
            first_row = seed[:1] if len(seed) > 0 else avg_series_full[:1]
            pad = np.tile(first_row, (pad_len, 1))
            seed = np.concatenate([pad, seed], axis=0)
        this_horizon = min(reseed_every, total_horizon - step)
        center = seed.mean(axis=0).astype(np.float32) if local_recenter else layer_means
        chunk = simulate_dmd_trend(A_tilde, Ur, seed, embed_dim=embed_dim,
                                    horizon=this_horizon, layer_means=center)
        if prev_last is not None and crossfade_steps > 0:
            cf = min(crossfade_steps, this_horizon)
            for j in range(cf):
                w = (j + 1) / (cf + 1)
                chunk[j] = (1.0 - w) * prev_last + w * chunk[j]
        preds[step:step + this_horizon] = chunk
        prev_last = chunk[this_horizon - 1].copy()
        step += this_horizon
    return preds


avg_series_train = masked_layer_average_np(data_train_stats)
# v7.17 -- heuristic changed from `//6` to `*0.6`: with the 10/10/80 split,
# the train (formerly "stats", 50%) window is now 5x smaller -- using a
# LARGER fraction of what's available brings the resulting embed_dim closer
# to the ~600 steps (~ the real ~530-step oscillation period) found
# necessary, instead of only ~166 with the old heuristic.
DMD_EMBED_DIM = min(600, max(20, int(len(data_train_stats) * 0.6)))
print(f"Fitting DMD for trend conditioning (embed_dim={DMD_EMBED_DIM})...")
DMD_A, DMD_Ur, DMD_RANK, DMD_LAYER_MEANS = fit_dmd_trend(avg_series_train, embed_dim=DMD_EMBED_DIM)
print(f"Trend DMD fitted: rank={DMD_RANK} (out of embed_dim*9={DMD_EMBED_DIM*9})")

DMD_RESEED_EVERY = 25
avg_series_full = masked_layer_average_np(np.concatenate([data_train_stats, data_val, data_test], axis=0))
DMD_FORECAST_VALTEST = simulate_dmd_trend_rolling(
    DMD_A, DMD_Ur, avg_series_full, embed_dim=DMD_EMBED_DIM,
    start_idx=len(data_train_stats), total_horizon=len(data_val) + len(data_test),
    layer_means=DMD_LAYER_MEANS, reseed_every=DMD_RESEED_EVERY,
)
DMD_FORECAST_VAL = DMD_FORECAST_VALTEST[:len(data_val)]
DMD_FORECAST_TEST = DMD_FORECAST_VALTEST[len(data_val):]

CORE_MASK_NP = (data_train_stats[0] != 0).astype(np.float32)

_dev_train = data_train_stats - masked_layer_average_np(data_train_stats)[:, :, None, None]
_core_mask_b = CORE_MASK_NP[None].astype(bool)
TEXTURE_ENV_LO = np.zeros(9, dtype=np.float32)
TEXTURE_ENV_HI = np.zeros(9, dtype=np.float32)
TEXTURE_STD_TARGET = np.zeros(9, dtype=np.float32)
for _l in range(9):
    _devs_l = _dev_train[:, _l][np.broadcast_to(_core_mask_b[:, _l], _dev_train[:, _l].shape)]
    TEXTURE_ENV_LO[_l] = np.percentile(_devs_l, 0.05)
    TEXTURE_ENV_HI[_l] = np.percentile(_devs_l, 99.95)
    _mask_l = _core_mask_b[0, _l]
    _per_frame_std = _dev_train[:, _l][:, _mask_l].std(axis=1)
    TEXTURE_STD_TARGET[_l] = float(_per_frame_std.mean())

# v7.20 -- new: PER-PIXEL texture envelope (careful reimplementation of the
# v7.11 idea, which was reverted for a broadcasting bug -- this time with an
# explicit shape unit test before integration, see CHANGELOG.md v7.20).
# Motive/evidence: the whole-layer SCALAR envelope bounds deviation
# MAGNITUDE but cannot prohibit large deviations in LOCATIONS where reality
# never has them -- a real run showed a bright "plume" artifact at the SAME
# location (top edge) in predicted L0-L4 at t+1000, and corner blobs in L8
# at t+15, all within the allowed scalar range but physically impossible at
# that LOCATION according to training data. PER-PIXEL percentiles (along the
# training time axis) capture exactly that: at a top-edge pixel where the
# real deviation never exceeds +0.01, the per-pixel envelope caps it at
# ~+0.012 (with margin), making the plume impossible by construction -- like
# the other post-decode guarantees.
TEXTURE_ENV_LO_PX = np.percentile(_dev_train, 0.5, axis=0).astype(np.float32)    # (9, H, W)
TEXTURE_ENV_HI_PX = np.percentile(_dev_train, 99.5, axis=0).astype(np.float32)   # (9, H, W)
# Additive margin on top of the multiplicative one: with only ~10% of
# training data, per-pixel percentiles are noisier than the scalar ones --
# the additive margin (a fraction of the layer's typical std) avoids
# over-clamping pixels whose sample envelope came out artificially narrow.
TEXTURE_ENV_PX_MARGIN_MULT = 1.25
_px_margin_add = (0.25 * TEXTURE_STD_TARGET)[:, None, None]
TEXTURE_ENV_LO_PX = (TEXTURE_ENV_LO_PX * TEXTURE_ENV_PX_MARGIN_MULT - _px_margin_add) * CORE_MASK_NP
TEXTURE_ENV_HI_PX = (TEXTURE_ENV_HI_PX * TEXTURE_ENV_PX_MARGIN_MULT + _px_margin_add) * CORE_MASK_NP

# v7.26 -- new: SPATIAL SMOOTHING of the per-pixel envelope (bugfix for a
# real artifact, see CHANGELOG.md v7.26). Evidence: real runs showed a
# bright RING/donut in L3-L7 at long horizon (invisible at t+15, it
# accumulates with steps) -- and the flow field (v7.24/7.25) showed
# predictions becoming too circular where reality is elongated. Mechanical
# cause: the real hot band shifts slightly frame to frame during training --
# at pixels ever "within" the band's reach the percentile allows a wide
# range; at pixels never covered the range is narrow -- a HARD BOUNDARY
# exactly where the band's historical reach ends, applied at every step of
# a long rollout, accumulates until visible (the ring). Fix: spatially
# smooth `TEXTURE_ENV_LO_PX`/`HI_PX` with a MASK-AWARE Gaussian blur
# (normalized by the blurred mask, so the exterior doesn't contaminate the
# boundary -- tested before integration: a hard 0.01->0.10 transition at one
# pixel becomes a 0.033->0.077->0.096->0.077->0.033 gradient). This WIDENS
# the envelope in most cases (averaging with more permissive neighbors), so
# regression risk is low -- a wider envelope restricts LESS, never more.
def _smooth_masked_2d(arr, mask, sigma=0.8):
    mask_f = mask.astype(np.float64)
    num = gaussian_filter(arr.astype(np.float64) * mask_f, sigma=sigma, mode="constant", cval=0.0)
    den = gaussian_filter(mask_f, sigma=sigma, mode="constant", cval=0.0)
    out = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-6)
    return (out * mask_f).astype(np.float32)


TEXTURE_ENV_SMOOTH_SIGMA = 0.8
for _l in range(9):
    TEXTURE_ENV_LO_PX[_l] = _smooth_masked_2d(TEXTURE_ENV_LO_PX[_l], CORE_MASK_NP[_l], sigma=TEXTURE_ENV_SMOOTH_SIGMA)
    TEXTURE_ENV_HI_PX[_l] = _smooth_masked_2d(TEXTURE_ENV_HI_PX[_l], CORE_MASK_NP[_l], sigma=TEXTURE_ENV_SMOOTH_SIGMA)

# v7.30 -- new: CAP on per-pixel width relative to the layer's SCALAR width
# (see CHANGELOG.md v7.30). Evidence: a real run showed a bright, CONSTANT
# streak in one boundary column, present already from t+10 and NOT
# accumulating with horizon (unlike the v7.28 CoordConv bug, reverted in
# v7.29) -- consistent with a pixel whose historical per-pixel percentile
# came out anomalously wide (plausible at the boundary, with only ~10% of
# training data). No pixel's envelope width (hi-lo) may exceed
# `TEXTURE_ENV_PX_MAX_WIDTH_RATIO` times that layer's SCALAR width -- an
# ADDITIONAL restriction on the already-smoothed envelope that can only
# NARROW anomalous pixels, never widen any.
TEXTURE_ENV_PX_MAX_WIDTH_RATIO = 1.5
for _l in range(9):
    _scalar_width = float(TEXTURE_ENV_HI[_l] - TEXTURE_ENV_LO[_l])
    _max_width = TEXTURE_ENV_PX_MAX_WIDTH_RATIO * _scalar_width
    _px_width = TEXTURE_ENV_HI_PX[_l] - TEXTURE_ENV_LO_PX[_l]
    _center = 0.5 * (TEXTURE_ENV_HI_PX[_l] + TEXTURE_ENV_LO_PX[_l])
    _needs_cap = _px_width > _max_width
    TEXTURE_ENV_LO_PX[_l] = np.where(_needs_cap, _center - _max_width / 2, TEXTURE_ENV_LO_PX[_l])
    TEXTURE_ENV_HI_PX[_l] = np.where(_needs_cap, _center + _max_width / 2, TEXTURE_ENV_HI_PX[_l])
TEXTURE_ENV_LO_PX = (TEXTURE_ENV_LO_PX * CORE_MASK_NP).astype(np.float32)
TEXTURE_ENV_HI_PX = (TEXTURE_ENV_HI_PX * CORE_MASK_NP).astype(np.float32)

# v7.31 -- for BOUNDARY cells (adjacent to the mask exterior), an attempt
# was made to use the whole-LAYER scalar envelope instead of the per-pixel
# one.
# v7.32 -- CORRECTION: that idea was misdirected and made things WORSE (see
# CHANGELOG.md v7.32). Evidence: a real run showed the boundary artifact
# INTENSIFIED -- a geometric ring/frame pattern appeared inside the field,
# bright columns on BOTH edges instead of just one. Reason: the whole-layer
# scalar reflects the CENTER's variability (where the hot spot moves, much
# wider) -- using it at the boundary doesn't "tighten" the constraint there,
# it WIDENS it well beyond what a boundary pixel actually needs, letting it
# oscillate MORE freely than before.
#
# Correct fix: instead of a single pixel's percentile (too noisy, v7.20-
# v7.30) or the whole-layer scalar (too wide, v7.31), deviations from ONLY
# the boundary cells are POOLED -- together, over the entire statistics
# window -- and the percentile is computed on that pooled set. This gives a
# large sample count (all boundary cells x all frames), so it's
# statistically robust, BUT bounded to the boundary region's REAL
# variability, without inheriting the much larger center variability that
# mattered for the whole-layer scalar.
_boundary_mask = np.zeros((9, TARGET_HW[0], TARGET_HW[1]), dtype=bool)
TEXTURE_ENV_BOUNDARY_LO = np.zeros(9, dtype=np.float32)
TEXTURE_ENV_BOUNDARY_HI = np.zeros(9, dtype=np.float32)
for _l in range(9):
    _m = CORE_MASK_NP[_l] > 0
    _shift_up = np.zeros_like(_m); _shift_up[:-1, :] = _m[1:, :]
    _shift_down = np.zeros_like(_m); _shift_down[1:, :] = _m[:-1, :]
    _shift_left = np.zeros_like(_m); _shift_left[:, :-1] = _m[:, 1:]
    _shift_right = np.zeros_like(_m); _shift_right[:, 1:] = _m[:, :-1]
    _has_all_neighbors = _shift_up & _shift_down & _shift_left & _shift_right
    _boundary_mask[_l] = _m & (~_has_all_neighbors)
    if _boundary_mask[_l].sum() > 0:
        _pool = _dev_train[:, _l][:, _boundary_mask[_l]].ravel()
        TEXTURE_ENV_BOUNDARY_LO[_l] = float(np.percentile(_pool, 0.5)) * TEXTURE_ENV_PX_MARGIN_MULT
        TEXTURE_ENV_BOUNDARY_HI[_l] = float(np.percentile(_pool, 99.5)) * TEXTURE_ENV_PX_MARGIN_MULT
    else:
        TEXTURE_ENV_BOUNDARY_LO[_l] = TEXTURE_ENV_LO[_l]
        TEXTURE_ENV_BOUNDARY_HI[_l] = TEXTURE_ENV_HI[_l]
    TEXTURE_ENV_LO_PX[_l] = np.where(_boundary_mask[_l], TEXTURE_ENV_BOUNDARY_LO[_l], TEXTURE_ENV_LO_PX[_l])
    TEXTURE_ENV_HI_PX[_l] = np.where(_boundary_mask[_l], TEXTURE_ENV_BOUNDARY_HI[_l], TEXTURE_ENV_HI_PX[_l])
TEXTURE_ENV_LO_PX = (TEXTURE_ENV_LO_PX * CORE_MASK_NP).astype(np.float32)
TEXTURE_ENV_HI_PX = (TEXTURE_ENV_HI_PX * CORE_MASK_NP).astype(np.float32)
print(f"Boundary cells (POOLED boundary-region envelope, v7.32 -- corrects v7.31): "
      f"{[int(_boundary_mask[l].sum()) for l in range(9)]} of {[int((CORE_MASK_NP[l] > 0).sum()) for l in range(9)]} total cells per layer")
print("  pooled boundary width vs. whole-layer scalar width (should be SMALLER, not larger):")
print("   boundary:", np.round(TEXTURE_ENV_BOUNDARY_HI - TEXTURE_ENV_BOUNDARY_LO, 4))
print("   scalar:  ", np.round(TEXTURE_ENV_HI - TEXTURE_ENV_LO, 4))

del _dev_train, _devs_l
TEXTURE_ENV_MARGIN = 1.15
TEXTURE_STD_MAX_RATIO = 1.25
print("Per-layer texture envelope (deviation vs. layer average, normalized space):")
print("  lo:", np.round(TEXTURE_ENV_LO, 4))
print("  hi:", np.round(TEXTURE_ENV_HI, 4))
print("  typical spatial std:", np.round(TEXTURE_STD_TARGET, 4))
print("PER-PIXEL envelope (v7.20, smoothed in v7.26, width-capped in v7.30, "
      "boundary=pooled in v7.32): mean range per layer "
      f"lo={np.round([TEXTURE_ENV_LO_PX[l][CORE_MASK_NP[l] > 0].mean() for l in range(9)], 4)} "
      f"hi={np.round([TEXTURE_ENV_HI_PX[l][CORE_MASK_NP[l] > 0].mean() for l in range(9)], 4)} "
      f"(sigma={TEXTURE_ENV_SMOOTH_SIGMA}, max_width_ratio={TEXTURE_ENV_PX_MAX_WIDTH_RATIO})")


# ==============================================================================
# 1c. CENTROID/SPREAD MODEL -- v7.15, new
# ==============================================================================
# "Find parameters in the dataset, and start from there": from the
# statistics window (large, NO gradient -- same philosophy as the trend
# DMD and the texture envelope), we extract the time series of WHERE each
# layer's hot spot is and how spread/blurred it is, and fit a cheap linear
# forecaster on that series -- reusing EXACTLY the same Hankel-DMD machinery
# (`fit_dmd_trend`/`simulate_dmd_trend_rolling` are generic over any (T,n)
# series).
#
# Usage: available as `CENTROID_FORECAST_VAL`/`CENTROID_FORECAST_TEST` for
# future diagnostics/extension. The REAL improvement mechanism in this
# version is `centroid_spread_consistency_loss` (below), which during Phase
# 2.5 uses the real ground truth within the training window -- it doesn't
# need this forecast. It's kept computed and documented because it's
# exactly the piece the user asked for ("find parameters... and start from
# there"), and because it's useful to compare against observed centroid
# drift in long val/test rollouts in future diagnostic work.
def compute_centroid_spread_series_np(arr, mask):
    """arr: (T, 9, H, W) numpy. mask: (9, H, W) boolean (fixed core mask).
    Returns (T, 9, 3): [row_centroid, col_centroid, spread] per layer and
    frame -- same definition (weight = value - layer minimum in that mask,
    in THAT frame) as `compute_centroid_and_spread` in the diagnostics
    script (v7.12), vectorized here over the whole time axis."""
    T = arr.shape[0]
    H, W = arr.shape[2], arr.shape[3]
    rows_grid, cols_grid = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    out = np.zeros((T, 9, 3), dtype=np.float32)
    for layer in range(9):
        m = mask[layer]
        rows_m, cols_m = rows_grid[m].astype(np.float64), cols_grid[m].astype(np.float64)
        vals = arr[:, layer][:, m].astype(np.float64)              # (T, n_mask)
        w = vals - vals.min(axis=1, keepdims=True)
        w_sum = w.sum(axis=1)
        flat_mask = w_sum < 1e-8
        w_sum_safe = np.where(flat_mask, 1.0, w_sum)
        r_c = (w * rows_m[None, :]).sum(axis=1) / w_sum_safe
        c_c = (w * cols_m[None, :]).sum(axis=1) / w_sum_safe
        spread = np.sqrt(((w * ((rows_m[None, :] - r_c[:, None]) ** 2
                                 + (cols_m[None, :] - c_c[:, None]) ** 2)).sum(axis=1)) / w_sum_safe)
        if flat_mask.any():
            r_c[flat_mask] = rows_m.mean()
            c_c[flat_mask] = cols_m.mean()
            spread[flat_mask] = 0.0
        out[:, layer, 0] = r_c
        out[:, layer, 1] = c_c
        out[:, layer, 2] = spread
    return out


centroid_spread_train = compute_centroid_spread_series_np(data_train_stats, CORE_MASK_NP.astype(bool))
centroid_spread_train_flat = centroid_spread_train.reshape(len(data_train_stats), 27)
CENTROID_EMBED_DIM_EFF = min(CENTROID_EMBED_DIM, max(20, int(len(data_train_stats) * 0.6)))
print(f"Fitting centroid/spread model (embed_dim={CENTROID_EMBED_DIM_EFF})...")
try:
    CTR_A, CTR_Ur, CTR_RANK, CTR_MEANS = fit_dmd_trend(centroid_spread_train_flat, embed_dim=CENTROID_EMBED_DIM_EFF)
    print(f"Centroid/spread model fitted: rank={CTR_RANK}")
    centroid_spread_full = compute_centroid_spread_series_np(
        np.concatenate([data_train_stats, data_val, data_test], axis=0), CORE_MASK_NP.astype(bool)
    ).reshape(-1, 27)
    CENTROID_FORECAST_VALTEST = simulate_dmd_trend_rolling(
        CTR_A, CTR_Ur, centroid_spread_full, embed_dim=CENTROID_EMBED_DIM_EFF,
        start_idx=len(data_train_stats), total_horizon=len(data_val) + len(data_test),
        layer_means=CTR_MEANS, reseed_every=CENTROID_RESEED_EVERY,
    ).reshape(-1, 9, 3)
    CENTROID_FORECAST_VAL = CENTROID_FORECAST_VALTEST[:len(data_val)]
    CENTROID_FORECAST_TEST = CENTROID_FORECAST_VALTEST[len(data_val):]
except Exception as e:
    print(f"[!] Could not fit the centroid/spread model ({e}) -- "
          f"continuing without it (only affects future diagnostics/extension, not "
          f"the training loss, which uses real ground truth within the window).")
    CENTROID_FORECAST_VAL = None
    CENTROID_FORECAST_TEST = None

# v7.18 -- new: per-layer CONFIDENCE weight for
# `centroid_spread_consistency_loss`. Motivation (real regression in v7.17:
# Phase 2.5 exploded to NaN -- the centroid loss started at ~8.5 and kept
# rising, ~100x larger than in previous runs). Diagnosis: not every layer
# has a well-defined "hot spot" -- the most turbulent layer has
# dispersed/ambiguous behavior (multiple foci shifting frame to frame), so
# its "real" centroid is itself NOISY (jumps a lot step to step even for a
# physically valid field) -- forcing the model to chase a target that is
# basically noise is not a useful physical signal, and destabilizes
# training. Layers with genuinely clearer movement/dissipation (a real
# centroid that shifts smoothly and slowly) DO give a useful signal.
#
# The variance of the real centroid's STEP (frame to frame) is measured per
# layer, over the training window -- a centroid that jumps a lot step to
# step (high step variance) is the signature of "no general focal point,"
# exactly what the user described. The resulting weight is LOWER for those
# layers (so as not to chase noise) and HIGHER for layers with a stable
# centroid (to actually use that useful signal).
try:
    _centroid_step = np.diff(centroid_spread_train[:, :, :2], axis=0)          # (T-1, 9, 2): (d_row, d_col) per step
    _centroid_step_var = (_centroid_step ** 2).sum(axis=-1).mean(axis=0)        # (9,): mean squared step size
    CENTROID_LAYER_WEIGHT = 1.0 / (1.0 + _centroid_step_var / (_centroid_step_var.mean() + 1e-8))
    CENTROID_LAYER_WEIGHT = (CENTROID_LAYER_WEIGHT / CENTROID_LAYER_WEIGHT.max()).astype(np.float32)
except Exception as e:
    print(f"[!] Could not compute the per-layer centroid weight ({e}) -- using a uniform weight (1.0).")
    CENTROID_LAYER_WEIGHT = np.ones(9, dtype=np.float32)
print("Per-layer confidence weight for the centroid/spread loss "
      "(1.0 = stable, useful centroid movement; near 0 = layer without a "
      "clear dense point, centroid dominated by noise -- see CHANGELOG.md v7.18):")
print(" ", np.round(CENTROID_LAYER_WEIGHT, 3))


# ==============================================================================
# 1d. GEOMETRIC-SYMMETRY DATA AUGMENTATION -- v7.15, new
# ==============================================================================
def _dihedral_transforms():
    """The 8 transformations of the D4 dihedral group over a tensor
    (..., H, W). Operate on the last two dimensions regardless of how many
    batch/time/layer dimensions precede them."""
    return {
        "identity":       lambda t: t,
        "rot90":          lambda t: torch.rot90(t, 1, dims=(-2, -1)),
        "rot180":         lambda t: torch.rot90(t, 2, dims=(-2, -1)),
        "rot270":         lambda t: torch.rot90(t, 3, dims=(-2, -1)),
        "flip_h":         lambda t: torch.flip(t, dims=(-1,)),
        "flip_v":         lambda t: torch.flip(t, dims=(-2,)),
        "transpose":      lambda t: torch.transpose(t, -2, -1),
        "anti_transpose": lambda t: torch.flip(torch.transpose(t, -2, -1), dims=(-2, -1)),
    }


def find_valid_symmetries(core_mask_np, atol=1e-6):
    """Tests every dihedral-group transform against the FIXED core mask and
    keeps only those that leave it EXACTLY unchanged -- programmatic
    verification, not an assumption. A 4-loop PWR reactor may have
    rotation/reflection symmetry in the core layout; this confirms it
    against real data instead of assuming it. Always includes at least the
    identity."""
    mask_t = torch.tensor(core_mask_np, dtype=torch.float32)
    transforms = _dihedral_transforms()
    valid = []
    for name, fn in transforms.items():
        try:
            transformed = fn(mask_t)
            if transformed.shape == mask_t.shape and torch.allclose(transformed, mask_t, atol=atol):
                valid.append((name, fn))
        except RuntimeError:
            continue
    if not valid:
        valid = [("identity", transforms["identity"])]
    names = [n for n, _ in valid]
    print(f"Valid symmetries for data augmentation (leave the core mask exactly unchanged): {names}")
    return valid


def apply_symmetry_to_window(window, fn):
    """window: (..., 9, H, W). The SAME transform is applied to ALL layers
    and ALL frames of the window/sequence at once -- the temporal
    relationship between frames must be preserved. The per-layer spatial
    average (`trend`) is invariant under these transforms, so it doesn't
    need to be recomputed after augmenting."""
    return fn(window)


def maybe_jitter_noise(window, std=NOISE_JITTER_STD, prob=NOISE_JITTER_PROB):
    """Small Gaussian noise, only inside the mask (cells != 0 of the tensor
    itself), with probability `prob` per sample. `std` is small relative to
    the real typical spatial std (`TEXTURE_STD_TARGET`, ~0.05-0.18) so as
    not to introduce unrealistic texture."""
    if np.random.rand() > prob:
        return window
    mask = (window != 0.0).float()
    noise = torch.randn_like(window) * std * mask
    return window + noise


VALID_SYMMETRIES = find_valid_symmetries(CORE_MASK_NP) if USE_SYMMETRY_AUGMENTATION else \
    [("identity", _dihedral_transforms()["identity"])]


def _maybe_apply_symmetry(window):
    if USE_SYMMETRY_AUGMENTATION and np.random.rand() < SYMMETRY_AUGMENT_PROB and len(VALID_SYMMETRIES) > 1:
        _, fn = VALID_SYMMETRIES[np.random.randint(len(VALID_SYMMETRIES))]
        return apply_symmetry_to_window(window, fn)
    return window


# ==============================================================================
# 2. DATASETS
# ==============================================================================
class FrameDatasetMF(Dataset):
    """Individual frames + fidelity id -- Phase 1 (autoencoder).
    v7.15: optional `augment` (noise jitter; the frame is both input and
    reconstruction target, so noise acts as denoising-autoencoder-style
    regularization, standard and low-risk).

    v7.16 -- IMPORTANT: this class NEVER applies symmetry augmentation
    anymore, regardless of `USE_SYMMETRY_AUGMENTATION` (it used to). See
    CHANGELOG.md v7.16: a real run confirmed that rotating/flipping single
    frames without giving the decoder any pose signal confuses the
    autoencoder (reconstruction Val MAPE went from ~10% to ~17%, plus a
    cross-shaped artifact in long-horizon spatial predictions) -- the
    decoder ended up "averaging" the 8 orientations instead of
    reconstructing a single one. Only noise jitter is kept (much smaller,
    no evidence it causes harm)."""

    def __init__(self, arr, fidelity_id, augment=False):
        self.X = torch.tensor(arr, dtype=torch.float32)
        self.fid = fidelity_id
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment and USE_NOISE_JITTER:
            x = maybe_jitter_noise(x)
        return x, torch.tensor(self.fid, dtype=torch.long)


class NextFrameDatasetMF(Dataset):
    """(window of SEQ_LEN frames, next frame, fidelity, next frame's trend)
    -- Phase 2. v7.15: symmetry is applied to CONTEXT + TARGET together
    (concatenated, so the transform is consistent across both); noise
    jitter ONLY on the context (`x_seq`), never on the target (`x_next`), so
    as not to corrupt the supervision signal."""

    def __init__(self, arr, fidelity_id, seq_len=SEQ_LEN, augment=False):
        self.X = torch.tensor(arr, dtype=torch.float32)
        self.fid = fidelity_id
        self.seq_len = seq_len
        self.trend = torch.tensor(masked_layer_average_np(arr), dtype=torch.float32)
        self.augment = augment

    def __len__(self):
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.seq_len]
        x_next = self.X[idx + self.seq_len]
        trend_next = self.trend[idx + self.seq_len]
        if self.augment:
            combined = torch.cat([x_seq, x_next.unsqueeze(0)], dim=0)
            combined = _maybe_apply_symmetry(combined)
            x_seq, x_next = combined[:self.seq_len], combined[self.seq_len]
            if USE_NOISE_JITTER:
                x_seq = maybe_jitter_noise(x_seq)
            # trend_next is a spatial average -> invariant under D4, no recomputation needed
        return x_seq, x_next, torch.tensor(self.fid, dtype=torch.long), trend_next


class RolloutDatasetMF(Dataset):
    """(initial window, K real future frames, fidelity, per-step trend) --
    Phase 2.5 and 3. v7.15: symmetry over the whole window (context+future
    together, temporal consistency); noise jitter ONLY on the context
    `x_seq`, never on `y_seq` (the Phase 2.5 multi-step loss target).

    v7.19 -- new: `stride`. With stride=1 (default, high fidelity) ALL
    windows are kept. With stride>1 (low/medium, abundant) one window every
    `stride` steps is taken -- two windows 1 step apart share ~99.7% of
    content at k_steps=320, so the stride removes near-pure redundancy and
    cuts per-epoch cost ~stride-fold without sacrificing temporal coverage
    (see CHANGELOG.md v7.19)."""

    def __init__(self, arr, fidelity_id, seq_len=SEQ_LEN, k_steps=K_ROLLOUT_STEPS, augment=False, stride=1):
        self.X = torch.tensor(arr, dtype=torch.float32)
        self.fid = fidelity_id
        self.seq_len = seq_len
        self.k_steps = k_steps
        self.total = seq_len + k_steps
        self.stride = max(1, int(stride))
        self.trend = torch.tensor(masked_layer_average_np(arr), dtype=torch.float32)
        self.augment = augment

    def __len__(self):
        n_full = max(0, len(self.X) - self.total + 1)
        return (n_full + self.stride - 1) // self.stride

    def __getitem__(self, idx):
        idx = idx * self.stride
        window = self.X[idx: idx + self.total]
        trend_seq = self.trend[idx + self.seq_len: idx + self.total]
        if self.augment:
            window = _maybe_apply_symmetry(window)
        x_seq = window[: self.seq_len]
        y_seq = window[self.seq_len:]
        if self.augment and USE_NOISE_JITTER:
            x_seq = maybe_jitter_noise(x_seq)
        return x_seq, y_seq, torch.tensor(self.fid, dtype=torch.long), trend_seq


class ReactorWindowDataset(Dataset):
    """Classic 2-tuple format (x_seq, y_seq), used for final evaluation --
    NEVER augmented (val/test/probe must always be real, untransformed
    data)."""

    def __init__(self, arr, seq_len=SEQ_LEN, horizon_len=20, trend_override=None):
        self.X = torch.tensor(arr, dtype=torch.float32)
        self.seq_len = seq_len
        self.horizon_len = horizon_len
        self.total = seq_len + horizon_len
        trend_np = trend_override if trend_override is not None else masked_layer_average_np(arr)
        self.trend = torch.tensor(trend_np, dtype=torch.float32)

    def __len__(self):
        return max(0, len(self.X) - self.total + 1)

    def __getitem__(self, idx):
        window = self.X[idx: idx + self.total]
        return window[: self.seq_len], window[self.seq_len:]

    def get_trend_window(self, idx):
        return self.trend[idx + self.seq_len: idx + self.total]


def make_loaders(k_steps=K_ROLLOUT_STEPS):
    # v7.15 -- augment=True on the THREE training loaders of each fidelity
    # (low, medium, high) -- the user explicitly asked for augmentation on
    # all three. val/test/probe are ALWAYS unaugmented.
    # v7.16 -- reminder: for `FrameDatasetMF` (Phase 1), `augment=True` now
    # ONLY activates noise jitter (symmetry was removed from that class, see
    # CHANGELOG.md v7.16); for `NextFrameDatasetMF`/`RolloutDatasetMF`
    # (Phase 2/2.5), `augment=True` respects `USE_SYMMETRY_AUGMENTATION`
    # (False by default) in addition to jitter.
    loader_low_ae = DataLoader(FrameDatasetMF(data_low_n, FIDELITY_ID["low"], augment=True),
                                 batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    loader_medium_ae = DataLoader(FrameDatasetMF(data_medium_n, FIDELITY_ID["medium"], augment=True),
                                  batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    loader_high_ae = DataLoader(FrameDatasetMF(data_train, FIDELITY_ID["high"], augment=True),
                                 batch_size=min(BATCH_SIZE, max(1, len(data_train))), shuffle=True)

    loader_low_dyn = DataLoader(NextFrameDatasetMF(data_low_n, FIDELITY_ID["low"], augment=True),
                                  batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    loader_medium_dyn = DataLoader(NextFrameDatasetMF(data_medium_n, FIDELITY_ID["medium"], augment=True),
                                   batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    loader_high_dyn = DataLoader(NextFrameDatasetMF(data_train, FIDELITY_ID["high"], augment=True),
                                  batch_size=min(BATCH_SIZE, max(1, len(data_train) - SEQ_LEN)), shuffle=True)

    ds_rollout_high = RolloutDatasetMF(data_train, FIDELITY_ID["high"], k_steps=k_steps, augment=True)
    loader_rollout_high = DataLoader(ds_rollout_high, batch_size=min(BATCH_SIZE, max(1, len(ds_rollout_high))), shuffle=True)

    # v7.18 -- new: rollout loaders for low/medium fidelity -- Phase 2.5a
    # (rollout-consistency pretraining on abundant data, see CHANGELOG.md
    # v7.18). Same `k_steps` as the high-fidelity one so the horizon
    # curriculum is consistent between 2.5a and 2.5b.
    # v7.19 -- stride=ROLLOUT_WINDOW_STRIDE on low/medium (cost control, see
    # CHANGELOG.md v7.19); high fidelity keeps stride=1 (no windows dropped
    # from the scarce 10%).
    ds_rollout_low = RolloutDatasetMF(data_low_n, FIDELITY_ID["low"], k_steps=k_steps, augment=True,
                                        stride=ROLLOUT_WINDOW_STRIDE)
    ds_rollout_medium = RolloutDatasetMF(data_medium_n, FIDELITY_ID["medium"], k_steps=k_steps, augment=True,
                                         stride=ROLLOUT_WINDOW_STRIDE)
    loader_rollout_low = DataLoader(ds_rollout_low, batch_size=min(BATCH_SIZE, max(1, len(ds_rollout_low))), shuffle=True)
    loader_rollout_medium = DataLoader(ds_rollout_medium, batch_size=min(BATCH_SIZE, max(1, len(ds_rollout_medium))), shuffle=True)
    print(f"Phase 2.5a: rollout windows after stride={ROLLOUT_WINDOW_STRIDE} -> "
          f"low={len(ds_rollout_low)}, medium={len(ds_rollout_medium)} "
          f"(additional cap: {MAX_ROLLOUT_BATCHES_PER_EPOCH} batches/fidelity/epoch)")

    # v7.17 -- horizon_len raised 20 -> CHECKPOINT_EVAL_HORIZON (200): this
    # `val_ds` is used by `quick_rollout_val_mape` to pick the best Phase
    # 2.5 checkpoint -- it needs to see beyond 20 steps for that selection
    # to reflect the phase's real objective (see CHANGELOG.md v7.17).
    val_ds = ReactorWindowDataset(data_val, horizon_len=min(CHECKPOINT_EVAL_HORIZON, max(1, len(data_val) - SEQ_LEN)),
                                   trend_override=DMD_FORECAST_VAL)
    test_ds = ReactorWindowDataset(data_test, horizon_len=min(20, max(1, len(data_test) - SEQ_LEN)),
                                    trend_override=DMD_FORECAST_TEST)
    val_loader_ae = DataLoader(FrameDatasetMF(data_val, FIDELITY_ID["high"]), batch_size=min(BATCH_SIZE, max(1, len(data_val))), shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=min(BATCH_SIZE, max(1, len(test_ds))), shuffle=False, drop_last=False)

    probe_horizon_available = max(1, len(data_val) - SEQ_LEN)
    probe_ds = ReactorWindowDataset(data_val, horizon_len=min(PROBE_HORIZON, probe_horizon_available),
                                     trend_override=DMD_FORECAST_VAL)

    return dict(
        loader_low_ae=loader_low_ae, loader_medium_ae=loader_medium_ae, loader_high_ae=loader_high_ae,
        loader_low_dyn=loader_low_dyn, loader_medium_dyn=loader_medium_dyn, loader_high_dyn=loader_high_dyn,
        loader_rollout_high=loader_rollout_high,
        loader_rollout_low=loader_rollout_low, loader_rollout_medium=loader_rollout_medium,
        val_ds=val_ds, test_ds=test_ds, val_loader_ae=val_loader_ae,
        test_loader=test_loader, probe_ds=probe_ds,
    )


# ==============================================================================
# 3. SHARED BLOCKS (fidelity, conv encoder, decoder) -- NO architecture changes
# ==============================================================================
class FidelityEmbedding(nn.Module):
    def __init__(self, n_fidelities=N_FIDELITIES, embed_dim=FIDELITY_DIM):
        super().__init__()
        self.embed = nn.Embedding(n_fidelities, embed_dim)

    def forward(self, fidelity_id):
        return self.embed(fidelity_id)


class ConvFeatureExtractor(nn.Module):
    def __init__(self, out_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(9, 16, 3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, out_ch, 3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(4, out_ch), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class FrameEncoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, feat_ch=32, n_fidelities=N_FIDELITIES, fidelity_dim=FIDELITY_DIM):
        super().__init__()
        self.feat = ConvFeatureExtractor(feat_ch)
        self.pool = nn.AdaptiveAvgPool2d((3, 3))
        flat_dim = feat_ch * 3 * 3
        self.fidelity_embed = FidelityEmbedding(n_fidelities, fidelity_dim)
        self.to_latent = nn.Sequential(
            nn.Linear(flat_dim + fidelity_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh(),
        )

    def forward(self, x_frame, fidelity_id):
        f = self.feat(x_frame)
        f = self.pool(f).flatten(1)
        fid = self.fidelity_embed(fidelity_id)
        return self.to_latent(torch.cat([f, fid], dim=-1))

    def encode_sequence(self, x_seq, fidelity_id):
        B, T = x_seq.shape[0], x_seq.shape[1]
        x_flat = x_seq.reshape(B * T, *x_seq.shape[2:])
        fid_rep = fidelity_id.repeat_interleave(T)
        z_flat = self.forward(x_flat, fid_rep)
        return z_flat.reshape(B, T, -1)


class BoundedDecoder(nn.Module):
    """v7.29 -- REVERTED to the pre-v7.28 architecture (see CHANGELOG.md
    v7.29: the real run showed a SEVERE regression -- L6/L7 collapsed to a
    nearly flat field with a saturated column at the boundary, worsening
    with horizon -- caused precisely by CoordConv). Back to bilinear
    upsampling + 3x3 convolutions from a 4x4 grid, with NO coordinate
    channels and no residual smoothing block. The curvature loss (v7.28,
    `spatial_curvature_loss`) IS kept -- it's a loss mechanism, not an
    architecture change, and doesn't share the risk that caused this
    regression.

    v7.30 -- `padding_mode` changed from "replicate" to "reflect" in the two
    convolutions (see CHANGELOG.md v7.30). Evidence: after the CoordConv
    revert, a boundary streak persisted -- but CONSTANT from t+10 (not
    accumulating with horizon like the v7.28 bug), the signature of a
    different bias: "replicate" padding repeats the boundary value outward,
    giving the convolution an artificially "flat" edge statistic that can
    systematically bias those cells. "reflect" (mirrors the real gradient
    instead of flattening it) is the standard alternative for this specific
    bias -- a single hyperparameter change on an ALREADY-EXISTING layer, no
    new capacity or parameters, much lower risk than CoordConv."""

    def __init__(self, latent_dim=LATENT_DIM, n_fidelities=N_FIDELITIES, fidelity_dim=FIDELITY_DIM,
                 max_delta=MAX_DECODER_DELTA):
        super().__init__()
        self.max_delta = max_delta
        self.fidelity_embed = FidelityEmbedding(n_fidelities, fidelity_dim)
        self.in_proj = nn.Linear(latent_dim + fidelity_dim, latent_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64 * 4 * 4),
            nn.Unflatten(1, (64, 4, 4)),
            nn.Upsample(size=(8, 8), mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1, padding_mode="reflect"),
            nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Upsample(size=(15, 15), mode="bilinear", align_corners=False),
            nn.Conv2d(32, 9, 3, padding=1, padding_mode="reflect"),
        )

    def forward(self, z, fidelity_id, x_prev=None, core_mask=None):
        fid = self.fidelity_embed(fidelity_id)
        z_cond = self.in_proj(torch.cat([z, fid], dim=-1))
        raw_out = self.net(z_cond)

        if x_prev is None:
            out = torch.sigmoid(raw_out)
        else:
            delta = torch.tanh(raw_out) * self.max_delta
            out = (x_prev + delta).clamp(0.0, 1.0)

        if core_mask is not None:
            out = out * core_mask.unsqueeze(0)
        return out


class PhysicsCoefficients(nn.Module):
    def __init__(self, n_layers=9, learnable=True, v_z_init=1.0, alpha_init=0.05, source_init=0.0):
        super().__init__()
        if learnable:
            self.log_v_z = nn.Parameter(torch.tensor(float(np.log(v_z_init + 1e-4))))
            self.log_alpha = nn.Parameter(torch.tensor(float(np.log(alpha_init + 1e-4))))
            self.source = nn.Parameter(torch.ones(n_layers) * source_init)
        else:
            self.register_buffer("log_v_z", torch.tensor(float(np.log(v_z_init + 1e-4))))
            self.register_buffer("log_alpha", torch.tensor(float(np.log(alpha_init + 1e-4))))
            self.register_buffer("source", torch.ones(n_layers) * source_init)

    @property
    def v_z(self):
        return torch.exp(self.log_v_z)

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    def report(self):
        with torch.no_grad():
            print(f"v_z (effective axial velocity) = {self.v_z.item():.5f}")
            print(f"alpha (in-plane diffusivity)    = {self.alpha.item():.5f}")
            print(f"S_i (per-layer source, 0..8)     = {self.source.cpu().numpy().round(4)}")


# ==============================================================================
# 4. DETERMINISTIC DYNAMICS -- CAUSAL TRANSFORMER WITH SLIDING WINDOW
# ==============================================================================
class TrendConditioner(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, trend_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(trend_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh(),
        )

    def forward(self, trend_vec):
        return self.net(trend_vec)


class LatentTransformerDynamics(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, seq_len=SEQ_LEN, n_heads=4, n_layers=3, ff_dim=256):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.pos_embed = nn.Embedding(seq_len + 1, latent_dim)
        self.query_token = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        self.trend_conditioner = TrendConditioner(latent_dim=latent_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=n_heads, dim_feedforward=ff_dim,
            batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_proj = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, latent_dim))

    def forward(self, z_buffer, trend_vec=None):
        B, T, D = z_buffer.shape
        dev = z_buffer.device
        pos_ids = torch.arange(T, device=dev)
        tokens = z_buffer + self.pos_embed(pos_ids)[None]
        q_pos = torch.full((1,), T, device=dev, dtype=torch.long)
        query = self.query_token.expand(B, -1, -1) + self.pos_embed(q_pos)[None]
        if trend_vec is None:
            trend_vec = torch.zeros(B, 9, device=dev, dtype=z_buffer.dtype)
        query = query + self.trend_conditioner(trend_vec).unsqueeze(1)
        seq = torch.cat([tokens, query], dim=1)
        out = self.transformer(seq)
        return self.out_proj(out[:, -1])


# ==============================================================================
# 5. GENERATIVE RESIDUAL -- RECTIFIED FLOW / FLOW MATCHING
# ==============================================================================
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        dev = t.device
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=dev) * (-math.log(10000) / max(half - 1, 1)))
        args = (t * 1000.0)[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class VelocityNet(nn.Module):
    def __init__(self, latent_dim, cond_dim, time_dim=64, hidden=256):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim + time_dim, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z_t, t, cond):
        te = self.time_emb(t)
        return self.net(torch.cat([z_t, cond, te], dim=-1))


class RectifiedFlowDynamics(nn.Module):
    def __init__(self, latent_dim, cond_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.velocity = VelocityNet(latent_dim, cond_dim)

    def compute_loss(self, z1_true, cond):
        B = z1_true.shape[0]
        dev = z1_true.device
        z0 = torch.randn_like(z1_true)
        t = torch.rand(B, device=dev)
        zt = (1 - t.unsqueeze(-1)) * z0 + t.unsqueeze(-1) * z1_true
        target_v = z1_true - z0
        pred_v = self.velocity(zt, t, cond)
        return F.mse_loss(pred_v, target_v)

    def sample(self, cond, n_steps=N_FLOW_STEPS, stochastic=False, noise_scale=0.0,
               guidance_fn=None, guidance_scale=0.0):
        B = cond.shape[0]
        dev = cond.device
        z = torch.randn(B, self.latent_dim, device=dev)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=dev)
            if guidance_fn is not None and guidance_scale > 0:
                with torch.enable_grad():
                    z_req = z.detach().requires_grad_(True)
                    v = self.velocity(z_req, t, cond)
                    g = guidance_fn(z_req, t)
                v = v.detach() + guidance_scale * g.detach()
            else:
                v = self.velocity(z, t, cond)
            z = z + v * dt
            if stochastic and i < n_steps - 1:
                z = z + noise_scale * math.sqrt(dt) * torch.randn_like(z)
        return z


# ==============================================================================
# 6. FULL MODEL
# ==============================================================================
class LatentWorldModelV6(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, seq_len=SEQ_LEN, n_fidelities=N_FIDELITIES):
        super().__init__()
        self.encoder = FrameEncoder(latent_dim, n_fidelities=n_fidelities)
        self.dynamics_mean = LatentTransformerDynamics(latent_dim, seq_len=seq_len)
        self.dynamics_flow = RectifiedFlowDynamics(latent_dim, cond_dim=latent_dim * 2)
        self.decoder = BoundedDecoder(latent_dim, n_fidelities=n_fidelities)
        self.physics = PhysicsCoefficients(n_layers=9)
        self.seq_len = seq_len
        self.n_fidelities = n_fidelities
        self.register_buffer("core_mask", torch.tensor(CORE_MASK_NP, dtype=torch.float32))
        self.register_buffer("texture_env_lo", torch.tensor(TEXTURE_ENV_LO * TEXTURE_ENV_MARGIN, dtype=torch.float32))
        self.register_buffer("texture_env_hi", torch.tensor(TEXTURE_ENV_HI * TEXTURE_ENV_MARGIN, dtype=torch.float32))
        self.register_buffer("texture_std_target", torch.tensor(TEXTURE_STD_TARGET, dtype=torch.float32))
        # v7.18 -- new: see CHANGELOG.md v7.18 (Phase 2.5 NaN fix)
        self.register_buffer("centroid_layer_weight", torch.tensor(CENTROID_LAYER_WEIGHT, dtype=torch.float32))
        # v7.20 -- new: per-pixel envelope (see CHANGELOG.md v7.20)
        self.register_buffer("texture_env_lo_px", torch.tensor(TEXTURE_ENV_LO_PX, dtype=torch.float32))
        self.register_buffer("texture_env_hi_px", torch.tensor(TEXTURE_ENV_HI_PX, dtype=torch.float32))

    def apply_texture_std_renorm(self, x_field, max_ratio=TEXTURE_STD_MAX_RATIO):
        mask = self.core_mask.unsqueeze(0)
        counts = mask.sum(dim=(-2, -1)).clamp(min=1)
        mean = ((x_field * mask).sum(dim=(-2, -1)) / counts).unsqueeze(-1).unsqueeze(-1)
        dev = (x_field - mean) * mask
        var = (dev ** 2).sum(dim=(-2, -1)) / counts
        std = torch.sqrt(var + 1e-12)
        target = self.texture_std_target.view(1, 9) * max_ratio
        scale = torch.clamp(target / (std + 1e-12), max=1.0)
        return (mean + dev * scale.unsqueeze(-1).unsqueeze(-1)) * mask

    def _default_fidelity(self, B, dev):
        return torch.full((B,), self.n_fidelities - 1, dtype=torch.long, device=dev)

    def apply_texture_envelope(self, x_field):
        mask = self.core_mask.unsqueeze(0)
        counts = mask.sum(dim=(-2, -1)).clamp(min=1)
        mean = ((x_field * mask).sum(dim=(-2, -1)) / counts).unsqueeze(-1).unsqueeze(-1)
        dev = x_field - mean
        lo = self.texture_env_lo.view(1, 9, 1, 1)
        hi = self.texture_env_hi.view(1, 9, 1, 1)
        dev_clamped = torch.clamp(dev, min=lo, max=hi)
        return (mean + dev_clamped) * mask

    def apply_texture_envelope_px(self, x_field):
        """v7.20 -- new: PER-PIXEL envelope (see CHANGELOG.md v7.20). Same as
        `apply_texture_envelope` but with lo/hi bounds of shape (9,H,W) --
        each pixel is clamped to the range of deviations THAT pixel showed
        during training (with margin), prohibiting by construction localized
        artifacts (plumes/blobs in locations reality never has) that the
        per-layer scalar envelope can't touch. Reimplementation of the v7.11
        idea, now with broadcasting explicitly unit-tested before
        integration: x_field is (B,9,H,W); lo/hi (9,H,W) -> unsqueeze(0) ->
        (1,9,H,W); torch.clamp with same-rank tensors broadcasts
        element-wise over B unambiguously. Applied AFTER the scalar envelope
        (the scalar bounds global magnitude; this one bounds location)."""
        mask = self.core_mask.unsqueeze(0)
        counts = mask.sum(dim=(-2, -1)).clamp(min=1)
        mean = ((x_field * mask).sum(dim=(-2, -1)) / counts).unsqueeze(-1).unsqueeze(-1)
        dev = x_field - mean
        lo = self.texture_env_lo_px.unsqueeze(0)   # (1, 9, H, W)
        hi = self.texture_env_hi_px.unsqueeze(0)   # (1, 9, H, W)
        dev_clamped = torch.clamp(dev, min=lo, max=hi)
        return (mean + dev_clamped) * mask

    def apply_trend_renorm(self, x_field, trend_vec):
        mask = self.core_mask.unsqueeze(0)
        counts = mask.sum(dim=(-2, -1)).clamp(min=1)
        current_mean = (x_field * mask).sum(dim=(-2, -1)) / counts
        shift = (trend_vec - current_mean).unsqueeze(-1).unsqueeze(-1)
        return x_field + shift * mask

    def encode_window(self, x_seq, fidelity_id=None):
        if fidelity_id is None:
            fidelity_id = self._default_fidelity(x_seq.shape[0], x_seq.device)
        return self.encoder.encode_sequence(x_seq, fidelity_id)

    def predict_next_latent(self, z_buffer, n_flow_steps=N_FLOW_STEPS, stochastic=False,
                             guidance_fn=None, guidance_scale=0.0, trend_vec=None):
        z_mean = self.dynamics_mean(z_buffer, trend_vec=trend_vec)
        cond = torch.cat([z_mean, z_buffer[:, -1]], dim=-1)
        z_next = self.dynamics_flow.sample(
            cond, n_steps=n_flow_steps, stochastic=stochastic,
            guidance_fn=guidance_fn, guidance_scale=guidance_scale,
        )
        return z_next, z_mean

    def rollout(self, x_seq, n_steps, fidelity_id=None, n_flow_steps=N_FLOW_STEPS,
                stochastic=False, use_physics_guidance=False, guidance_scale=2.0, trend_seq=None):
        if fidelity_id is None:
            fidelity_id = self._default_fidelity(x_seq.shape[0], x_seq.device)
        z_buffer = self.encode_window(x_seq, fidelity_id)
        x_prev = x_seq[:, -1]
        preds = []
        for t in range(n_steps):
            guidance_fn = None
            if use_physics_guidance:
                guidance_fn = make_physics_guidance(self, x_prev, fidelity_id, self.physics, dt=DT_PINN)
            trend_vec = trend_seq[:, t] if trend_seq is not None else None
            z_next, _ = self.predict_next_latent(
                z_buffer, n_flow_steps=n_flow_steps, stochastic=stochastic,
                guidance_fn=guidance_fn, guidance_scale=guidance_scale, trend_vec=trend_vec,
            )
            x_next = self.decoder(z_next, fidelity_id, x_prev=x_prev, core_mask=self.core_mask)
            x_next = self.apply_texture_std_renorm(x_next)
            x_next = self.apply_texture_envelope(x_next)
            x_next = self.apply_texture_envelope_px(x_next)   # v7.20
            if trend_vec is not None:
                x_next = self.apply_trend_renorm(x_next, trend_vec)
            preds.append(x_next.unsqueeze(1))
            z_buffer = torch.cat([z_buffer[:, 1:], z_next.unsqueeze(1)], dim=1)
            x_prev = x_next
        return torch.cat(preds, dim=1)

    def rollout_mean_only(self, x_seq, n_steps, fidelity_id=None, trend_seq=None):
        if fidelity_id is None:
            fidelity_id = self._default_fidelity(x_seq.shape[0], x_seq.device)
        z_buffer = self.encode_window(x_seq, fidelity_id)
        x_prev = x_seq[:, -1]
        preds = []
        for t in range(n_steps):
            trend_vec = trend_seq[:, t] if trend_seq is not None else None
            z_mean = self.dynamics_mean(z_buffer, trend_vec=trend_vec)
            x_next = self.decoder(z_mean, fidelity_id, x_prev=x_prev, core_mask=self.core_mask)
            x_next = self.apply_texture_std_renorm(x_next)
            x_next = self.apply_texture_envelope(x_next)
            x_next = self.apply_texture_envelope_px(x_next)   # v7.20
            if trend_vec is not None:
                x_next = self.apply_trend_renorm(x_next, trend_vec)
            preds.append(x_next.unsqueeze(1))
            z_buffer = torch.cat([z_buffer[:, 1:], z_mean.unsqueeze(1)], dim=1)
            x_prev = x_next
        return torch.cat(preds, dim=1)


def make_physics_guidance(model, x_prev_frame, fidelity_id, physics, dt=1.0):
    def guidance_fn(z_t, t):
        x_preview = model.decoder(z_t, fidelity_id, x_prev=x_prev_frame, core_mask=model.core_mask)
        residual = single_step_pinn_residual(x_prev_frame, x_preview, physics, dt=dt)
        grad_z = torch.autograd.grad(residual, z_t, create_graph=False)[0]
        return -grad_z

    return guidance_fn


# ==============================================================================
# 7. LOSSES
# ==============================================================================
def masked_mse(pred, target):
    mask = target != 0.0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return F.mse_loss(pred[mask], target[mask])


def masked_mape(pred, target):
    mask = target != 0.0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return (torch.abs((target[mask] - pred[mask]) / target[mask]).mean() * 100)


def masked_gradient_loss(pred, target):
    def diff_and_mask(a, b, axis):
        a1 = a.narrow(axis, 1, a.shape[axis] - 1)
        a0 = a.narrow(axis, 0, a.shape[axis] - 1)
        b1 = b.narrow(axis, 1, b.shape[axis] - 1)
        b0 = b.narrow(axis, 0, b.shape[axis] - 1)
        mask = (b1 != 0.0) & (b0 != 0.0)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=a.device)
        return F.mse_loss((a1 - a0)[mask], (b1 - b0)[mask])

    return diff_and_mask(pred, target, axis=-1) + diff_and_mask(pred, target, axis=-2)


def masked_spatial_mean_series(x):
    mask = (x != 0.0).float()
    summed = (x * mask).sum(dim=(-2, -1))
    counts = mask.sum(dim=(-2, -1)).clamp(min=1.0)
    return summed / counts


def masked_spatial_std_series(x):
    mask = (x != 0.0).float()
    counts = mask.sum(dim=(-2, -1)).clamp(min=1.0)
    mean = (x * mask).sum(dim=(-2, -1)) / counts
    mean_sq = (x ** 2 * mask).sum(dim=(-2, -1)) / counts
    var = (mean_sq - mean ** 2).clamp(min=0.0)
    return torch.sqrt(var + 1e-8)


def spatial_variance_growth_penalty(pred_series, target_series, max_ratio=SPATIAL_VAR_MAX_RATIO):
    pred_std_sp = masked_spatial_std_series(pred_series)
    target_std_sp = masked_spatial_std_series(target_series)
    excess = F.relu(pred_std_sp - max_ratio * target_std_sp)
    return (excess ** 2).mean()


def temporal_level_and_spread_loss(pred_series, target_series):
    pred_lm = masked_spatial_mean_series(pred_series)
    target_lm = masked_spatial_mean_series(target_series)
    level_loss = F.mse_loss(pred_lm.mean(dim=1), target_lm.mean(dim=1))
    if pred_lm.shape[1] > 1:
        spread_loss = F.mse_loss(pred_lm.std(dim=1), target_lm.std(dim=1))
    else:
        spread_loss = torch.tensor(0.0, device=pred_series.device)
    return level_loss, spread_loss


def temporal_shape_loss(pred_series, target_series, eps=1e-6):
    pred_lm = masked_spatial_mean_series(pred_series)
    target_lm = masked_spatial_mean_series(target_series)
    pred_c = pred_lm - pred_lm.mean(dim=1, keepdim=True)
    target_c = target_lm - target_lm.mean(dim=1, keepdim=True)
    num = (pred_c * target_c).sum(dim=1)
    den = torch.sqrt((pred_c ** 2).sum(dim=1) + eps) * torch.sqrt((target_c ** 2).sum(dim=1) + eps)
    corr = num / den
    return (1.0 - corr).mean()


def spectral_shape_loss(pred_series, target_series):
    pred_lm = masked_spatial_mean_series(pred_series)
    target_lm = masked_spatial_mean_series(target_series)
    if pred_lm.shape[1] < 4:
        return torch.tensor(0.0, device=pred_series.device)
    pred_c = pred_lm - pred_lm.mean(dim=1, keepdim=True)
    target_c = target_lm - target_lm.mean(dim=1, keepdim=True)
    pred_mag = torch.fft.rfft(pred_c, dim=1).abs()
    target_mag = torch.fft.rfft(target_c, dim=1).abs()
    return F.mse_loss(pred_mag, target_mag)


def variance_floor_loss(pred_series, target_series, min_ratio=VAR_FLOOR_RATIO):
    pred_lm = masked_spatial_mean_series(pred_series)
    target_lm = masked_spatial_mean_series(target_series)
    if pred_lm.shape[1] < 2:
        return torch.tensor(0.0, device=pred_series.device)
    pred_std = pred_lm.std(dim=1)
    target_std = target_lm.std(dim=1)
    deficit = F.relu(min_ratio * target_std - pred_std)
    return (deficit ** 2).mean()


def variance_ceiling_loss(pred_series, target_series, max_ratio=VAR_CEIL_RATIO):
    """v7.21 -- new: symmetric counterpart to `variance_floor_loss`.
    Penalizes the temporal std of the per-layer mean series EXCEEDING
    `max_ratio` times the real one -- attacks the residual jitter (~1.5-2x
    after the v7.20 DMD fix) that the floor, by construction, cannot
    touch."""
    pred_lm = masked_spatial_mean_series(pred_series)
    target_lm = masked_spatial_mean_series(target_series)
    if pred_lm.shape[1] < 2:
        return torch.tensor(0.0, device=pred_series.device)
    pred_std = pred_lm.std(dim=1)
    target_std = target_lm.std(dim=1)
    excess = F.relu(pred_std - max_ratio * target_std)
    return (excess ** 2).mean()


def variance_growth_penalty(pred_series, target_series, max_growth_ratio=MAX_GROWTH_RATIO):
    pred_lm = masked_spatial_mean_series(pred_series)
    target_lm = masked_spatial_mean_series(target_series)
    H = pred_lm.shape[1]
    if H < 4:
        return torch.tensor(0.0, device=pred_series.device)
    half = H // 2
    eps = 1e-4
    pred_std_early = pred_lm[:, :half].std(dim=1)
    pred_std_late = pred_lm[:, half:].std(dim=1)
    target_std_early = target_lm[:, :half].std(dim=1)
    target_std_late = target_lm[:, half:].std(dim=1)
    pred_log_growth = torch.log(pred_std_late + eps) - torch.log(pred_std_early + eps)
    target_log_growth = torch.log(target_std_late + eps) - torch.log(target_std_early + eps)
    excess = F.relu(pred_log_growth - target_log_growth - math.log(max_growth_ratio))
    return (excess ** 2).mean()


# ------------------------------------------------------------------------------
# v7.15 -- new: concentration/diffusion physics (centroid and spread) +
# SPATIAL spectral loss (analog of the temporal one from v6.1).
# ------------------------------------------------------------------------------
def compute_centroid_spread_torch(x_field, core_mask):
    """Differentiable (PyTorch) version of `compute_centroid_and_spread`
    (defined in the diagnostics script, v7.12), vectorized over any number
    of leading batch/time dimensions. x_field: (..., 9, H, W). core_mask:
    the model's fixed (9,H,W) buffer. Returns (r_c, c_c, spread), each of
    shape (..., 9)."""
    orig_shape = x_field.shape
    H, W = orig_shape[-2], orig_shape[-1]
    n_layers = orig_shape[-3]
    x_flat = x_field.reshape(-1, n_layers, H, W)
    device, dtype = x_flat.device, x_flat.dtype

    mask_b = core_mask.to(dtype).unsqueeze(0)                        # (1,9,H,W)
    mask_bool = mask_b.bool().expand_as(x_flat)
    rows = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1)
    cols = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, W)

    masked_for_min = torch.where(mask_bool, x_flat, torch.full_like(x_flat, float("inf")))
    layer_min = masked_for_min.amin(dim=(-2, -1), keepdim=True).detach()   # (N,9,1,1)

    w = (x_flat - layer_min) * mask_b                                   # (N,9,H,W) >= 0 inside the mask
    w_sum = w.sum(dim=(-2, -1)).clamp(min=1e-8)                          # (N,9)

    r_c = (w * rows).sum(dim=(-2, -1)) / w_sum                          # (N,9)
    c_c = (w * cols).sum(dim=(-2, -1)) / w_sum                          # (N,9)

    N = x_flat.shape[0]
    dr2 = (rows - r_c.reshape(N, n_layers, 1, 1)) ** 2
    dc2 = (cols - c_c.reshape(N, n_layers, 1, 1)) ** 2
    spread = torch.sqrt((w * (dr2 + dc2)).sum(dim=(-2, -1)) / w_sum + 1e-8)   # (N,9)

    new_shape = orig_shape[:-3] + (n_layers,)
    return r_c.reshape(new_shape), c_c.reshape(new_shape), spread.reshape(new_shape)


def centroid_spread_consistency_loss(pred_series, target_series, core_mask, layer_weight=None,
                                      spread_weight=CENTROID_SPREAD_SUBWEIGHT):
    """v7.15 -- new. Penalizes the distance between the intensity-weighted
    centroid (hot-spot location) and the spread difference (effective
    radius) between prediction and the REAL future, within the training
    window itself -- doesn't need any external forecast, `target_series` IS
    already ground truth. Directly attacks what
    `plot_centroid_tracking`/`plot_transition_trigger_analysis` (v7.12/v7.13)
    could only diagnose: the feature drifting from its real position and
    blurring at long horizons. Conservative default weight (`W_CENTROID`) --
    see the v7.7 lesson in CHANGELOG.md before raising it.

    v7.18 -- new: `layer_weight` (9,), typically
    `model.centroid_layer_weight`. Motivation (critical bugfix -- see
    CHANGELOG.md v7.18): without this weight, a real run made this loss
    explode (~8.5 from epoch 1, rising to NaN) because it treated all
    layers equally -- including the most turbulent one, whose "real"
    centroid is itself noisy (no single, stable dense point there). The
    per-layer weight (computed from the data itself, see the data-loading
    section) lowers those layers' contribution to near zero and preserves
    the useful signal from layers with clearer movement/dissipation."""
    pr, pc, ps = compute_centroid_spread_torch(pred_series, core_mask)
    tr, tc, ts = compute_centroid_spread_torch(target_series, core_mask)
    centroid_dist2 = (pr - tr) ** 2 + (pc - tc) ** 2
    spread_diff2 = (ps - ts) ** 2
    per_layer_loss = centroid_dist2 + spread_weight * spread_diff2   # (..., 9)
    if layer_weight is not None:
        w_shape = [1] * (per_layer_loss.dim() - 1) + [9]
        per_layer_loss = per_layer_loss * layer_weight.view(*w_shape)
    return per_layer_loss.mean()


def spatial_spectral_loss(pred_field, target_field, core_mask):
    """v7.15 -- new. Analog of `spectral_shape_loss` (v6.1, TEMPORAL axis)
    but applied to the SPATIAL (H,W) plane of each frame: compares the 2D
    FFT magnitude of the deviation from the layer average. A uniform blob
    has ~0 spatial spectral energy at non-zero frequencies -- same
    well-conditioned-gradient mechanism that already worked for temporal
    collapse, applied here to TEXTURE collapse (L4-L8 in the real v7.14
    run). Computed on the DEVIATION (field minus layer average), not the
    raw field, so as not to penalize LEVEL differences (already covered by
    `apply_trend_renorm`/`loss_level`)."""
    orig_shape = pred_field.shape
    H, W = orig_shape[-2], orig_shape[-1]
    n_layers = orig_shape[-3]
    pf = pred_field.reshape(-1, n_layers, H, W)
    tf = target_field.reshape(-1, n_layers, H, W)
    m = core_mask.to(pf.dtype).unsqueeze(0)                              # (1,9,H,W)
    counts = m.sum(dim=(-2, -1)).clamp(min=1).reshape(1, n_layers, 1, 1)
    pf_mean = (pf * m).sum(dim=(-2, -1), keepdim=True) / counts
    tf_mean = (tf * m).sum(dim=(-2, -1), keepdim=True) / counts
    pf_dev = (pf - pf_mean) * m
    tf_dev = (tf - tf_mean) * m
    pf_spec = torch.fft.rfft2(pf_dev, dim=(-2, -1)).abs()
    tf_spec = torch.fft.rfft2(tf_dev, dim=(-2, -1)).abs()
    return F.mse_loss(pf_spec, tf_spec)


def spatial_curvature_loss(pred, target):
    """v7.28 -- new. Penalizes the local CURVATURE (discrete 5-point
    Laplacian) of the prediction differing from reality -- directly attacks
    "block"/mosaic transitions (a physically smooth field has bounded local
    curvature; a hard patch boundary produces a curvature spike reality
    doesn't have). Same masking pattern as `masked_gradient_loss`: the
    5-point stencil is only evaluated where ALL points are non-zero in the
    target (avoids mixing with the mask exterior)."""
    def lap(x):
        center = x[..., 1:-1, 1:-1]
        left = x[..., 1:-1, :-2]
        right = x[..., 1:-1, 2:]
        up = x[..., :-2, 1:-1]
        down = x[..., 2:, 1:-1]
        return left + right + up + down - 4 * center

    t_center = target[..., 1:-1, 1:-1]
    t_left = target[..., 1:-1, :-2]
    t_right = target[..., 1:-1, 2:]
    t_up = target[..., :-2, 1:-1]
    t_down = target[..., 2:, 1:-1]
    valid = (t_center != 0) & (t_left != 0) & (t_right != 0) & (t_up != 0) & (t_down != 0)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    lap_p, lap_t = lap(pred), lap(target)
    return F.mse_loss(lap_p[valid], lap_t[valid])


def spatial_pattern_correlation_loss(pred_field, target_field, core_mask, eps=1e-8):
    """v7.22 -- new. 1 - Pearson correlation between predicted and real
    DEVIATION fields (field minus layer average), pixel by pixel inside the
    mask, averaged over batch/time/layer. Unlike `spatial_spectral_loss`
    (FFT magnitude, translation-invariant -- see CHANGELOG.md v7.22), this
    loss is POSITION-SENSITIVE: the same pattern shifted gives low
    correlation. The spatial analog of `temporal_shape_loss` (the temporal
    loss that has worked best throughout the project). Operating on
    deviations, it doesn't penalize LEVEL differences (covered by trend
    renorm / loss_level)."""
    orig_shape = pred_field.shape
    H, W = orig_shape[-2], orig_shape[-1]
    n_layers = orig_shape[-3]
    pf = pred_field.reshape(-1, n_layers, H * W)
    tf = target_field.reshape(-1, n_layers, H * W)
    m = core_mask.to(pf.dtype).reshape(1, n_layers, H * W)
    counts = m.sum(dim=-1).clamp(min=1.0)
    pf_mean = (pf * m).sum(dim=-1, keepdim=True) / counts.unsqueeze(-1)
    tf_mean = (tf * m).sum(dim=-1, keepdim=True) / counts.unsqueeze(-1)
    pf_dev = (pf - pf_mean) * m
    tf_dev = (tf - tf_mean) * m
    num = (pf_dev * tf_dev).sum(dim=-1)
    den = torch.sqrt((pf_dev ** 2).sum(dim=-1) + eps) * torch.sqrt((tf_dev ** 2).sum(dim=-1) + eps)
    corr = num / den
    return (1.0 - corr).mean()


def diffusion_consistency_loss(pred_series, target_series, alpha, layer_weight=None, dt=1.0):
    """v7.33 -- new (user's idea: "a fluid/thermal loss... see how each
    point reacts with its neighbor"). Residual of the PURE DIFFUSION
    equation (dC/dt = alpha*Laplacian(C)), WITHOUT the inter-layer axial
    term (that relationship is already covered by
    `interlayer_coupling_loss`, v7.24) -- exactly "how each point reacts
    with its 4 immediate neighbors" in a layer's plane, applied equally to
    all 9 layers.

    Unlike Phase 3 (PINN, optional, historically never fit well -- v7.4:
    "the residual never went to zero"), this version does NOT require the
    prediction's residual to be ZERO (that would assume pure diffusion with
    a fixed alpha exactly explains the real dynamics, something the
    project's own history already showed doesn't hold). Instead, the SAME
    residual is computed for prediction AND reality, and their DIFFERENCE is
    penalized -- only requiring the prediction to have the SAME degree of
    (in)consistency with a diffusive process that reality has. Much safer,
    same pattern as `interlayer_coupling_loss` (compare a quantity computed
    the same way on both, don't impose an absolute value).

    `layer_weight` (optional, typically `model.centroid_layer_weight`): the
    user themselves noted "it wouldn't apply much on the first layer
    either" -- L0 is the most turbulent, with less coherent/noisier
    behavior, so forcing diffusive consistency there with the same weight
    as smooth layers would be chasing noise (same reasoning that already
    protected `centroid_spread_consistency_loss` from a NaN collapse in
    v7.18). The same data-derived per-layer confidence weight is reused.

    Reuses `model.physics.alpha` (already calibrated if
    `RUN_PHYSICS_CALIBRATION` ran) -- already-validated infrastructure, no
    new mechanisms to test from scratch. Unit-tested before integration: ~0
    on a field that genuinely follows the diffusion equation compared
    against itself, clearly positive against random noise, finite
    gradients, and the per-layer weight correctly scales each layer's
    contribution."""
    def _residual(x):
        dCdt_full = (x[:, 1:] - x[:, :-1]) / dt
        C = x[:, :-1]
        center = C[..., 1:-1, 1:-1]
        left = C[..., 1:-1, :-2]
        right = C[..., 1:-1, 2:]
        up = C[..., :-2, 1:-1]
        down = C[..., 2:, 1:-1]
        lap = left + right + up + down - 4 * center
        dCdt = dCdt_full[..., 1:-1, 1:-1]
        residual = dCdt - alpha * lap
        valid = (center != 0) & (left != 0) & (right != 0) & (up != 0) & (down != 0)
        return residual, valid

    if pred_series.shape[1] < 2:
        return torch.tensor(0.0, device=pred_series.device)
    res_p, valid_p = _residual(pred_series)
    res_t, valid_t = _residual(target_series)
    valid = (valid_p & valid_t).float()
    diff2 = (res_p - res_t) ** 2 * valid
    counts = valid.sum(dim=(-2, -1)).clamp(min=1)
    per_layer = diff2.sum(dim=(-2, -1)) / counts   # (..., 9)
    if layer_weight is not None:
        w_shape = [1] * (per_layer.dim() - 1) + [9]
        per_layer = per_layer * layer_weight.view(*w_shape)
    return per_layer.mean()


def interlayer_coupling_loss(pred_field, target_field, core_mask, eps=1e-8):
    """v7.24 -- new (user's idea: "how lower layers affect upper ones").
    For each pair of ADJACENT layers (i, i+1) and each frame, computes the
    Pearson correlation between their DEVIATION fields (pattern of layer i
    vs. pattern of layer i+1, within the intersection of their masks) -- a
    direct measure of how much each layer "inherits" the pattern of the one
    below -- and penalizes (MSE) that coupling profile differing from the
    real one. The trainable/statistical counterpart to the axial coupling
    `pinn_residual_loss` imposes as a hard equation."""
    orig_shape = pred_field.shape
    H, W = orig_shape[-2], orig_shape[-1]
    n_layers = orig_shape[-3]
    pf = pred_field.reshape(-1, n_layers, H * W)
    tf = target_field.reshape(-1, n_layers, H * W)
    m = core_mask.to(pf.dtype).reshape(1, n_layers, H * W)
    counts = m.sum(dim=-1).clamp(min=1.0)
    pf_dev = (pf - (pf * m).sum(dim=-1, keepdim=True) / counts.unsqueeze(-1)) * m
    tf_dev = (tf - (tf * m).sum(dim=-1, keepdim=True) / counts.unsqueeze(-1)) * m

    def _pair_corr(dev):
        a = dev[:, :-1, :]          # layers 0..7
        b = dev[:, 1:, :]           # layers 1..8
        num = (a * b).sum(dim=-1)
        den = torch.sqrt((a ** 2).sum(dim=-1) + eps) * torch.sqrt((b ** 2).sum(dim=-1) + eps)
        return num / den            # (N, 8)

    return F.mse_loss(_pair_corr(pf_dev), _pair_corr(tf_dev))


def compute_shape_moments_torch(x_field, core_mask, eps=1e-8):
    """v7.25 -- new. Extends `compute_centroid_spread_torch` (which gives an
    ISOTROPIC radius, a circle) to the full second-moment tensor
    (intensity-weighted covariance ellipse) -- gives MAJOR radius, MINOR
    radius, and major-axis ORIENTATION. x_field: (..., 9, H, W). Returns
    (lambda_major, lambda_minor, cos2theta, sin2theta), each (..., 9).
    Orientation is returned as (cos(2*theta), sin(2*theta)) instead of theta
    directly -- an axis (unlike a vector) is indistinguishable from itself
    rotated 180 degrees, and the factor of 2 maps that ambiguity to a single
    continuous representation, avoiding wrap-around discontinuities in the
    loss."""
    orig_shape = x_field.shape
    H, W = orig_shape[-2], orig_shape[-1]
    n_layers = orig_shape[-3]
    x_flat = x_field.reshape(-1, n_layers, H, W)
    device, dtype = x_flat.device, x_flat.dtype

    mask_b = core_mask.to(dtype).unsqueeze(0)
    mask_bool = mask_b.bool().expand_as(x_flat)
    rows = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1)
    cols = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, W)

    masked_for_min = torch.where(mask_bool, x_flat, torch.full_like(x_flat, float("inf")))
    layer_min = masked_for_min.amin(dim=(-2, -1), keepdim=True).detach()
    w = (x_flat - layer_min) * mask_b
    w_sum = w.sum(dim=(-2, -1)).clamp(min=1e-8)

    r_c = (w * rows).sum(dim=(-2, -1)) / w_sum
    c_c = (w * cols).sum(dim=(-2, -1)) / w_sum
    N = x_flat.shape[0]
    dr = rows - r_c.reshape(N, n_layers, 1, 1)
    dc = cols - c_c.reshape(N, n_layers, 1, 1)

    Srr = (w * dr * dr).sum(dim=(-2, -1)) / w_sum
    Scc = (w * dc * dc).sum(dim=(-2, -1)) / w_sum
    Src = (w * dr * dc).sum(dim=(-2, -1)) / w_sum

    trace = Srr + Scc
    diff = Srr - Scc
    radius = torch.sqrt((diff * 0.5) ** 2 + Src ** 2 + eps)   # (S1-S2)/2, always >= 0
    lambda_major = trace * 0.5 + radius
    lambda_minor = (trace * 0.5 - radius).clamp(min=0.0)
    theta2_num = 2.0 * Src
    theta2_den = diff
    denom_norm = torch.sqrt(theta2_num ** 2 + theta2_den ** 2 + eps)
    cos2theta = theta2_den / denom_norm
    sin2theta = theta2_num / denom_norm

    new_shape = orig_shape[:-3] + (n_layers,)
    return (lambda_major.reshape(new_shape), lambda_minor.reshape(new_shape),
            cos2theta.reshape(new_shape), sin2theta.reshape(new_shape))


def spatial_anisotropy_loss(pred_series, target_series, core_mask, layer_weight=None):
    """v7.25 -- new. Compares SHAPE (major/minor radius ratio, i.e. how
    elongated the focus is) and ORIENTATION (which way the long axis points)
    between prediction and reality, via `compute_shape_moments_torch`.
    Evidence (see CHANGELOG.md v7.25): `plot_concentration_flow_field`
    showed L4 (a horizontal band in reality) becoming more radial/circular
    in the prediction at long horizon -- exactly an anisotropy loss that
    neither `centroid_spread_consistency_loss` (isotropic radius) nor
    `spatial_pattern_correlation_loss` (global correlation) penalize in a
    targeted way. Orientation is compared in (cos2theta, sin2theta) space --
    no wrap-around discontinuity, see the moments function.

    v7.30 -- new absolute-SCALE term on the major axis
    (`sqrt(lambda_major)`). Motivation (real evidence): matching only the
    major/minor RATIO doesn't prevent the focus from shrinking -- a 2x2 blob
    and a 10x2 band have different elongation ratios, but if the predicted
    ratio gets close enough to the real one the shape term is nearly
    satisfied even though the absolute SIZE (how far the band extends) is
    very wrong. This new term directly penalizes that scale difference,
    complementing (not replacing) the existing ratio+orientation term."""
    p_maj, p_min, p_c2, p_s2 = compute_shape_moments_torch(pred_series, core_mask)
    t_maj, t_min, t_c2, t_s2 = compute_shape_moments_torch(target_series, core_mask)
    eps = 1e-6
    elong_pred = p_maj / (p_min + eps)
    elong_target = t_maj / (t_min + eps)
    shape_term = (torch.log(elong_pred + 1.0) - torch.log(elong_target + 1.0)) ** 2
    orient_term = (p_c2 - t_c2) ** 2 + (p_s2 - t_s2) ** 2
    scale_term = (torch.sqrt(p_maj + eps) - torch.sqrt(t_maj + eps)) ** 2   # v7.30
    per_layer = shape_term + orient_term + scale_term
    if layer_weight is not None:
        w_shape = [1] * (per_layer.dim() - 1) + [9]
        per_layer = per_layer * layer_weight.view(*w_shape)
    return per_layer.mean()


def single_step_pinn_residual(x_prev, x_next, physics, dt=1.0, dz=1.0):
    dTdt_full = (x_next - x_prev) / dt
    T_t = x_prev
    center = T_t[..., 1:-1, 1:-1]
    left = T_t[..., 1:-1, :-2]
    right = T_t[..., 1:-1, 2:]
    up = T_t[..., :-2, 1:-1]
    down = T_t[..., 2:, 1:-1]
    lap = left + right + up + down - 4 * center
    dTdt = dTdt_full[..., 1:-1, 1:-1]
    prev_layer_full = torch.cat([T_t[:, :1], T_t[:, :-1]], dim=1)
    prev_layer = prev_layer_full[..., 1:-1, 1:-1]
    axial_term = physics.v_z * (center - prev_layer) / dz
    source = physics.source.view(1, -1, 1, 1)
    residual = dTdt + axial_term - physics.alpha * lap - source
    valid = (center != 0) & (left != 0) & (right != 0) & (up != 0) & (down != 0) & (prev_layer != 0)
    if valid.sum() == 0:
        return (x_next.sum() * 0.0)
    return (residual[valid] ** 2).mean()


def pinn_residual_loss(pred_series, physics, dt=1.0, dz=1.0):
    if pred_series.shape[1] < 2:
        return torch.tensor(0.0, device=pred_series.device)
    T_full = pred_series
    dTdt_full = (T_full[:, 1:] - T_full[:, :-1]) / dt
    T_t = T_full[:, :-1]
    center = T_t[..., 1:-1, 1:-1]
    left = T_t[..., 1:-1, :-2]
    right = T_t[..., 1:-1, 2:]
    up = T_t[..., :-2, 1:-1]
    down = T_t[..., 2:, 1:-1]
    lap = left + right + up + down - 4 * center
    dTdt = dTdt_full[..., 1:-1, 1:-1]
    prev_layer_full = torch.cat([T_t[:, :, :1], T_t[:, :, :-1]], dim=2)
    prev_layer = prev_layer_full[..., 1:-1, 1:-1]
    axial_term = physics.v_z * (center - prev_layer) / dz
    source = physics.source.view(1, 1, -1, 1, 1)
    residual = dTdt + axial_term - physics.alpha * lap - source
    valid = (center != 0) & (left != 0) & (right != 0) & (up != 0) & (down != 0) & (prev_layer != 0)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred_series.device)
    return (residual[valid] ** 2).mean()


# ==============================================================================
# 8. PHASE 1 -- PER-FRAME AUTOENCODER
# ==============================================================================
def train_step_autoencoder(model, x_batch, fid_batch, optimizer):
    """v7.15: adds `spatial_spectral_loss` (weight `W_SPATIAL_SPECTRAL_AE`)
    to fight texture collapse to a blob in small-physical-range layers when
    little fine-tuning data is available.
    v7.19: NaN/Inf guard -- a batch with non-finite loss is discarded
    without touching the weights (same safety net as Phase 2.5, see
    CHANGELOG.md v7.19). Returns the batch loss, or None if the batch was
    discarded."""
    optimizer.zero_grad()
    z = model.encoder(x_batch, fid_batch)
    x_hat = model.decoder(z, fid_batch, core_mask=model.core_mask)
    loss = (W_MSE * masked_mse(x_hat, x_batch)
            + W_GRAD * masked_gradient_loss(x_hat, x_batch)
            + W_SPATIAL_SPECTRAL_AE * spatial_spectral_loss(x_hat, x_batch, model.core_mask)
            + W_CURVATURE_AE * spatial_curvature_loss(x_hat, x_batch))
    if not torch.isfinite(loss):
        optimizer.zero_grad()
        return None
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(model.encoder.parameters()) + list(model.decoder.parameters()), GRAD_CLIP)
    optimizer.step()
    return loss.item()


def pretrain_autoencoder(model, loader_low, loader_medium, epochs=EPOCHS_AE_PRETRAIN):
    params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=LR_AE, weight_decay=1e-5)
    history = {"train_loss": []}
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for (x_b, f_b), (x_m, f_m) in zip(loader_low, loader_medium):
            for x, f in [(x_b, f_b), (x_m, f_m)]:
                x, f = x.to(device), f.to(device)
                step_loss = train_step_autoencoder(model, x, f, optimizer)
                if step_loss is None:   # v7.19: batch discarded for non-finite loss
                    continue
                total += step_loss
                n += 1
        avg = total / max(1, n)
        history["train_loss"].append(avg)
        print(f"[Phase 1a - AE pretraining] Epoch {epoch+1:03d}/{epochs} | Loss {avg:.5f}")
    return model, history


@torch.no_grad()
def evaluate_autoencoder_mape(model, loader):
    model.eval()
    total, n = 0.0, 0
    for x, f in loader:
        x, f = x.to(device), f.to(device)
        x_hat = model.decoder(model.encoder(x, f), f, core_mask=model.core_mask)
        total += masked_mape(x_hat, x).item()
        n += 1
    return total / max(1, n)


def finetune_autoencoder(model, loader_high, val_loader=None, epochs=EPOCHS_AE_FINETUNE,
                          patience=AE_FINETUNE_PATIENCE):
    """v7.15 -- new: early stopping + best checkpoint by Val MAPE. With a
    small `HIGH_FIDELITY_TRAIN_FRAC`, more epochs over the same data is
    cheap -- this allows raising `EPOCHS_AE_FINETUNE` without risking
    overfitting: the model state is saved every time Val MAPE improves, and
    training stops (restoring that best state) if it doesn't improve for
    `patience` consecutive epochs."""
    params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=LR_AE_FINETUNE, weight_decay=1e-5)
    history = {"train_loss": [], "val_mape": []}
    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for x, f in loader_high:
            x, f = x.to(device), f.to(device)
            step_loss = train_step_autoencoder(model, x, f, optimizer)
            if step_loss is None:   # v7.19: batch discarded for non-finite loss
                continue
            total += step_loss
            n += 1
        avg = total / max(1, n)
        history["train_loss"].append(avg)
        msg = f"[Phase 1b - AE fine-tuning, high fidelity] Epoch {epoch+1:03d}/{epochs} | Loss {avg:.5f}"
        if val_loader is not None and len(val_loader) > 0:
            val_mape = evaluate_autoencoder_mape(model, val_loader)
            history["val_mape"].append(val_mape)
            msg += f" | Val MAPE {val_mape:.3f}%"
            if val_mape < best_val - 1e-4:
                best_val = val_mape
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                msg += "  <- best so far"
            else:
                epochs_no_improve += 1
        print(msg)
        if val_loader is not None and epochs_no_improve >= patience:
            print(f"    [early stopping] no Val MAPE improvement in {patience} epochs -- "
                  f"stopping Phase 1b (best Val MAPE = {best_val:.3f}%).")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Phase 1b: restored the best checkpoint (Val MAPE = {best_val:.3f}%).")
    return model, history


# ==============================================================================
# 9. PHASE 2 -- ONE-STEP DYNAMICS (TRANSFORMER + FLOW MATCHING)
# ==============================================================================
def train_step_dynamics(model, x_seq, x_next, fid, optimizer, trend_next=None):
    optimizer.zero_grad()
    with torch.no_grad():
        z_buffer = model.encoder.encode_sequence(x_seq, fid)
        z_next_true = model.encoder(x_next, fid)
    z_mean = model.dynamics_mean(z_buffer, trend_vec=trend_next)
    loss_mean = F.mse_loss(z_mean, z_next_true)
    cond = torch.cat([z_mean, z_buffer[:, -1]], dim=-1)
    loss_flow = model.dynamics_flow.compute_loss(z_next_true, cond)
    loss = loss_mean + loss_flow
    if not torch.isfinite(loss):   # v7.19: batch discarded for non-finite loss
        optimizer.zero_grad()
        return None, None
    loss.backward()
    params = list(model.dynamics_mean.parameters()) + list(model.dynamics_flow.parameters())
    torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
    optimizer.step()
    return loss_mean.item(), loss_flow.item()


def pretrain_dynamics(model, loader_low, loader_medium, epochs=EPOCHS_DYN_PRETRAIN):
    params = list(model.dynamics_mean.parameters()) + list(model.dynamics_flow.parameters())
    optimizer = torch.optim.Adam(params, lr=LR_DYN)
    history = {"mse_mean": [], "flow_loss": []}
    for epoch in range(epochs):
        model.train()
        tot_m, tot_f, n = 0.0, 0.0, 0
        for (xb, yb, fb, tb), (xm, ym, fm, tm) in zip(loader_low, loader_medium):
            for x, y, f, tr in [(xb, yb, fb, tb), (xm, ym, fm, tm)]:
                x, y, f, tr = x.to(device), y.to(device), f.to(device), tr.to(device)
                lm, lf = train_step_dynamics(model, x, y, f, optimizer, trend_next=tr)
                if lm is None:   # v7.19: batch discarded for non-finite loss
                    continue
                tot_m += lm; tot_f += lf; n += 1
        history["mse_mean"].append(tot_m / max(1, n))
        history["flow_loss"].append(tot_f / max(1, n))
        print(f"[Phase 2a - dynamics pretraining] Epoch {epoch+1:03d}/{epochs} | "
              f"MSE_mean {tot_m/max(1,n):.5f} | FlowLoss {tot_f/max(1,n):.5f}")
    return model, history


def finetune_dynamics(model, loader_high, epochs=EPOCHS_DYN_FINETUNE):
    params = list(model.dynamics_mean.parameters()) + list(model.dynamics_flow.parameters())
    optimizer = torch.optim.Adam(params, lr=LR_DYN_FINETUNE)
    history = {"mse_mean": [], "flow_loss": []}
    for epoch in range(epochs):
        model.train()
        tot_m, tot_f, n = 0.0, 0.0, 0
        for x, y, f, tr in loader_high:
            x, y, f, tr = x.to(device), y.to(device), f.to(device), tr.to(device)
            lm, lf = train_step_dynamics(model, x, y, f, optimizer, trend_next=tr)
            if lm is None:   # v7.19: batch discarded for non-finite loss
                continue
            tot_m += lm; tot_f += lf; n += 1
        history["mse_mean"].append(tot_m / max(1, n))
        history["flow_loss"].append(tot_f / max(1, n))
        print(f"[Phase 2b - dynamics fine-tuning, high fidelity] Epoch {epoch+1:03d}/{epochs} | "
              f"MSE_mean {tot_m/max(1,n):.5f} | FlowLoss {tot_f/max(1,n):.5f}")
    return model, history


# ==============================================================================
# 10. PHASE 2.5 (NEW) -- ROLLOUT TRAINING (exposure-bias fix)
# ==============================================================================
def _rollout_chunk_mean_only(model, z_buffer, fid, chunk_len, noise_std=0.0, trend_chunk=None, x_prev_init=None):
    preds_decoded = []
    x_prev = x_prev_init
    for t in range(chunk_len):
        z_in = z_buffer + noise_std * torch.randn_like(z_buffer) if noise_std > 0 else z_buffer
        trend_vec = trend_chunk[:, t] if trend_chunk is not None else None
        z_mean = model.dynamics_mean(z_in, trend_vec=trend_vec)
        x_next = model.decoder(z_mean, fid, x_prev=x_prev, core_mask=model.core_mask)
        x_next = model.apply_texture_std_renorm(x_next)
        x_next = model.apply_texture_envelope(x_next)
        x_next = model.apply_texture_envelope_px(x_next)   # v7.20
        if trend_vec is not None:
            x_next = model.apply_trend_renorm(x_next, trend_vec)
        preds_decoded.append(x_next.unsqueeze(1))
        z_buffer = torch.cat([z_buffer[:, 1:], z_mean.unsqueeze(1)], dim=1)
        x_prev = x_next
    return torch.cat(preds_decoded, dim=1), z_buffer, x_prev


def train_step_rollout_consistency(model, x_seq, y_seq, fid, optimizer, k_steps,
                                    chunk_size=TBPTT_CHUNK, noise_std=NOISE_STD_ROLLOUT, trend_seq=None,
                                    spatial_loss_scale=1.0):
    """v7.15: adds `spatial_spectral_loss` and
    `centroid_spread_consistency_loss` (weights `W_SPATIAL_SPECTRAL`/
    `W_CENTROID`). Refactor: returns a DICTIONARY of totals (instead of a
    long positional tuple) so metrics can be extended without breaking the
    signature at every call site.

    v7.17 -- new: `spatial_loss_scale` (0..1) scales THE EFFECTIVE WEIGHT of
    those two losses only -- used by `finetune_rollout_consistency` for a
    warm-up ramp across epochs (see CHANGELOG.md v7.17: applying them at
    full strength from epoch 1 seems to have contributed to short-horizon
    MAPE worsening with more training)."""
    model.train()
    with torch.no_grad():
        z_buffer = model.encoder.encode_sequence(x_seq, fid)
    x_prev = x_seq[:, -1]

    n_chunks = math.ceil(k_steps / chunk_size)
    totals = {"loss": 0.0, "mse": 0.0, "shape": 0.0, "spectral": 0.0, "var_floor": 0.0,
              "var_ceil": 0.0,
              "growth": 0.0, "gradient": 0.0, "spatial_var": 0.0,
              "spatial_spectral": 0.0, "spatial_corr": 0.0, "interlayer": 0.0, "anisotropy": 0.0,
              "curvature": 0.0, "diffusion": 0.0, "centroid": 0.0}
    shape_by_chunk = []
    step_offset = 0
    early_ref = None
    growth_eps = 1e-4
    chunks_done = 0   # v7.18 -- new: only counts chunks that actually updated the weights
    for _ in range(n_chunks):
        this_chunk = min(chunk_size, k_steps - step_offset)
        optimizer.zero_grad()
        trend_chunk = trend_seq[:, step_offset:step_offset + this_chunk] if trend_seq is not None else None
        preds_decoded, z_buffer, x_prev = _rollout_chunk_mean_only(
            model, z_buffer, fid, this_chunk, noise_std=noise_std, trend_chunk=trend_chunk, x_prev_init=x_prev)
        y_chunk = y_seq[:, step_offset:step_offset + this_chunk]

        loss_mse = masked_mse(preds_decoded, y_chunk)
        loss_shape = temporal_shape_loss(preds_decoded, y_chunk)
        loss_spectral = spectral_shape_loss(preds_decoded, y_chunk)
        loss_var_floor = variance_floor_loss(preds_decoded, y_chunk)
        loss_var_ceil = variance_ceiling_loss(preds_decoded, y_chunk)   # v7.21
        loss_gradient = masked_gradient_loss(preds_decoded, y_chunk)
        loss_spatial_var = spatial_variance_growth_penalty(preds_decoded, y_chunk)
        loss_spatial_spectral = spatial_spectral_loss(preds_decoded, y_chunk, model.core_mask)
        loss_spatial_corr = spatial_pattern_correlation_loss(preds_decoded, y_chunk, model.core_mask)   # v7.22
        loss_interlayer = interlayer_coupling_loss(preds_decoded, y_chunk, model.core_mask)   # v7.24
        loss_anisotropy = spatial_anisotropy_loss(preds_decoded, y_chunk, model.core_mask,
                                                    layer_weight=model.centroid_layer_weight)   # v7.25
        loss_curvature = spatial_curvature_loss(preds_decoded, y_chunk)   # v7.28
        loss_diffusion = diffusion_consistency_loss(preds_decoded, y_chunk, model.physics.alpha.detach(),
                                                      layer_weight=model.centroid_layer_weight)   # v7.33
        loss_centroid = centroid_spread_consistency_loss(preds_decoded, y_chunk, model.core_mask,
                                                          layer_weight=model.centroid_layer_weight)
        level_loss, spread_loss = temporal_level_and_spread_loss(preds_decoded, y_chunk)
        loss = (loss_mse + W_SHAPE * loss_shape + W_LEVEL * level_loss + W_SPREAD * spread_loss
                + W_SPECTRAL * loss_spectral + W_VAR_FLOOR * loss_var_floor
                + W_VAR_CEIL * loss_var_ceil
                + W_GRAD_ROLLOUT * loss_gradient + W_SPATIAL_VAR * loss_spatial_var
                + spatial_loss_scale * (W_SPATIAL_SPECTRAL * loss_spatial_spectral
                                         + W_SPATIAL_CORR * loss_spatial_corr
                                         + W_INTERLAYER * loss_interlayer
                                         + W_ANISOTROPY * loss_anisotropy
                                         + W_CURVATURE * loss_curvature
                                         + W_DIFFUSION * loss_diffusion
                                         + W_CENTROID * loss_centroid))

        pred_lm_chunk = masked_spatial_mean_series(preds_decoded)
        target_lm_chunk = masked_spatial_mean_series(y_chunk)
        pred_log_std_chunk = torch.log(pred_lm_chunk.std(dim=1) + growth_eps)
        target_log_std_chunk = torch.log(target_lm_chunk.std(dim=1) + growth_eps)
        if early_ref is None:
            early_ref = (pred_log_std_chunk.detach(), target_log_std_chunk.detach())
            loss_growth = torch.tensor(0.0, device=preds_decoded.device)
        else:
            pred_log_std_ref, target_log_std_ref = early_ref
            pred_log_growth = pred_log_std_chunk - pred_log_std_ref
            target_log_growth = target_log_std_chunk - target_log_std_ref
            excess = F.relu(pred_log_growth - target_log_growth - math.log(MAX_GROWTH_RATIO))
            loss_growth = (excess ** 2).mean()
        loss = loss + W_GROWTH * loss_growth

        # v7.18 -- new: NaN/Inf safety guard (see CHANGELOG.md v7.18 -- a
        # real run exploded to NaN halfway through Phase 2.5, wasting the
        # rest of training). If this chunk's loss isn't finite, it's
        # discarded WITHOUT touching the weights or optimizer state, and the
        # rest of this batch's chunks are skipped (the forward pass that
        # produced it may already be corrupted -- chaining further chunks
        # would only make things worse).
        if not torch.isfinite(loss):
            print("    [!] Non-finite loss (NaN/Inf) in a Phase 2.5 chunk -- "
                  "discarding this batch WITHOUT modifying the weights (see CHANGELOG.md v7.18).")
            optimizer.zero_grad()
            break

        loss.backward()
        params = list(model.dynamics_mean.parameters()) + list(model.decoder.parameters())
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        optimizer.step()

        totals["loss"] += loss.item(); totals["mse"] += loss_mse.item()
        totals["shape"] += loss_shape.item(); totals["spectral"] += loss_spectral.item()
        totals["var_floor"] += loss_var_floor.item()
        totals["var_ceil"] += loss_var_ceil.item()
        totals["growth"] += loss_growth.item() if torch.is_tensor(loss_growth) else loss_growth
        totals["gradient"] += loss_gradient.item(); totals["spatial_var"] += loss_spatial_var.item()
        totals["spatial_spectral"] += loss_spatial_spectral.item()
        totals["spatial_corr"] += loss_spatial_corr.item()
        totals["interlayer"] += loss_interlayer.item()
        totals["anisotropy"] += loss_anisotropy.item()
        totals["curvature"] += loss_curvature.item()
        totals["diffusion"] += loss_diffusion.item()
        totals["centroid"] += loss_centroid.item()
        shape_by_chunk.append(loss_shape.item())
        chunks_done += 1

        z_buffer = z_buffer.detach()
        x_prev = x_prev.detach()
        step_offset += this_chunk

    for k in totals:
        totals[k] /= max(1, chunks_done)
    # v7.19 -- third return value: was there at least one non-finite chunk
    # in this batch? (used by the epoch-level rollback, see CHANGELOG.md v7.19)
    had_nonfinite = chunks_done < n_chunks
    return totals, shape_by_chunk, had_nonfinite


@torch.no_grad()
def probe_long_rollout_stability(model, dataset, horizon=PROBE_HORIZON,
                                  divergence_mape=PROBE_DIVERGENCE_MAPE, sample_idx=0):
    model.eval()
    if len(dataset) == 0:
        return None
    x_seq, y_seq = dataset[sample_idx]
    x_seq = x_seq.unsqueeze(0).to(device)
    horizon = min(horizon, y_seq.shape[0])
    if horizon < 2:
        return None
    y_seq = y_seq[:horizon].to(device)
    fid = model._default_fidelity(1, device)
    trend_seq = None
    if hasattr(dataset, "get_trend_window"):
        trend_seq = dataset.get_trend_window(sample_idx)[:horizon].unsqueeze(0).to(device)
    preds = model.rollout_mean_only(x_seq, horizon, fidelity_id=fid, trend_seq=trend_seq).squeeze(0)
    preds_real = denormalize_tensor(preds, v_min, v_max)
    target_real = denormalize_tensor(y_seq, v_min, v_max)

    divergence_step = None
    for h in range(horizon):
        mask = target_real[h] != 0.0
        if mask.sum() == 0:
            continue
        mape = (torch.abs((target_real[h][mask] - preds_real[h][mask]) / target_real[h][mask]).mean() * 100).item()
        if mape > divergence_mape:
            divergence_step = h
            break

    if divergence_step is not None:
        print(f"    [!] Stability probe: DIVERGES at t+{divergence_step + 1} "
              f"(MAPE > {divergence_mape:.0f}%) within a {horizon}-step rollout.")
    else:
        print(f"    Stability probe: no divergence in {horizon} steps "
              f"(MAPE < {divergence_mape:.0f}% across the whole horizon).")
    return divergence_step


def pretrain_rollout_consistency(model, loader_rollout_low, loader_rollout_medium,
                                  epochs=EPOCHS_ROLLOUT_PRETRAIN, k_steps=K_ROLLOUT_STEPS,
                                  k_steps_start=K_ROLLOUT_STEPS_START, chunk_size=TBPTT_CHUNK):
    """v7.18 -- new: PHASE 2.5a. Explicit user request: "low-fidelity models
    should also learn how each layer's physics behaves, so it can be
    learned more easily afterward with high-fidelity data." Until v7.17,
    ROLLOUT-CONSISTENCY training (multi-step, with TBPTT and the
    texture/centroid guarantees/losses) only ran on the 10% high-fidelity
    data -- Phase 2 (one step) did pretrain on low/medium, but Phase 2.5
    (full rollout) had no equivalent.

    This function fills that gap: it runs EXACTLY the same training
    mechanism as `finetune_rollout_consistency` (TBPTT, k_steps curriculum,
    same losses -- reuses `train_step_rollout_consistency` unchanged), but
    on low+medium fidelity (abundant, cheap to generate). The later
    fine-tuning on the 10% high-fidelity data (Phase 2.5b,
    `finetune_rollout_consistency`) then starts from an already-reasonable
    rollout dynamic, instead of having to learn rollout consistency from
    scratch with scarce data -- same pretrain/finetune pattern as Phase 1
    (1a/1b) and Phase 2 (2a/2b).

    No early stopping/checkpointing here (same as 1a/2a) -- careful
    checkpoint selection happens afterward, in Phase 2.5b, on real
    high-fidelity data, which is what actually matters to evaluate."""
    params = list(model.dynamics_mean.parameters()) + list(model.decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=LR_ROLLOUT_PRETRAIN)
    key_names = ["loss", "mse", "shape", "spectral", "var_floor", "var_ceil", "growth", "gradient",
                 "spatial_var", "spatial_spectral", "spatial_corr", "interlayer", "anisotropy",
                 "curvature", "diffusion", "centroid"]
    history = {("loss_total" if k == "loss" else f"loss_{k}"): [] for k in key_names}
    history["k_steps_epoch"] = []
    lr_backoffs = 0   # v7.19 -- counter of consecutive LR reductions (epoch rollback)

    for epoch in range(epochs):
        frac = epoch / max(1, epochs - 1)
        k_steps_epoch = int(round(k_steps_start + frac * (k_steps - k_steps_start)))
        k_steps_epoch = max(chunk_size, min(k_steps, k_steps_epoch))
        spatial_loss_scale = min(1.0, (epoch + 1) / max(1, SPATIAL_LOSS_WARMUP_EPOCHS))

        # v7.19 -- snapshot at the start of the epoch (rollback if it turns out corrupt)
        epoch_snapshot = copy.deepcopy(model.state_dict())
        model.train()
        running = {k: 0.0 for k in key_names}
        n = 0
        n_nonfinite = 0
        for (xb, yb, fb, trb), (xm, ym, fm, trm) in zip(loader_rollout_low, loader_rollout_medium):
            if n >= 2 * MAX_ROLLOUT_BATCHES_PER_EPOCH:   # v7.19 -- hard cap (2 fidelities per iteration)
                break
            for x, y, f, tr in [(xb, yb, fb, trb), (xm, ym, fm, trm)]:
                x, y, f, tr = x.to(device), y.to(device), f.to(device), tr.to(device)
                tr_epoch = tr[:, :k_steps_epoch]
                totals, _, had_nonfinite = train_step_rollout_consistency(
                    model, x, y, f, optimizer, k_steps=k_steps_epoch, chunk_size=chunk_size,
                    trend_seq=tr_epoch, spatial_loss_scale=spatial_loss_scale)
                if had_nonfinite:
                    n_nonfinite += 1
                for k in key_names:
                    running[k] += totals[k]
                n += 1

        # v7.19 -- epoch-level rollback + LR backoff (see CHANGELOG.md v7.19)
        nonfinite_frac = n_nonfinite / max(1, n)
        if nonfinite_frac >= NONFINITE_EPOCH_FRACTION:
            lr_backoffs += 1
            model.load_state_dict(epoch_snapshot)
            for g in optimizer.param_groups:
                g["lr"] = g["lr"] * 0.5
            print(f"[Phase 2.5a] Epoch {epoch+1:03d}: {n_nonfinite}/{n} batches with non-finite loss "
                  f"(>= {NONFINITE_EPOCH_FRACTION:.0%}) -- epoch DISCARDED (rollback to snapshot), "
                  f"LR reduced to {optimizer.param_groups[0]['lr']:.2e} "
                  f"(backoff {lr_backoffs}/{MAX_LR_BACKOFFS}).")
            if lr_backoffs >= MAX_LR_BACKOFFS:
                print("[Phase 2.5a] LR backoff limit reached -- stopping this phase, "
                      "keeping the last healthy state.")
                break
            continue
        lr_backoffs = 0   # healthy epoch -> reset the consecutive-backoff counter

        for k in key_names:
            hist_key = "loss_total" if k == "loss" else f"loss_{k}"
            history[hist_key].append(running[k] / max(1, n))
        history["k_steps_epoch"].append(k_steps_epoch)
        nf_note = f" | [!] {n_nonfinite}/{n} batches discarded (non-finite)" if n_nonfinite else ""
        print(f"[Phase 2.5a - rollout PREtraining, low+medium] Epoch {epoch+1:03d}/{epochs} | "
              f"k_steps {k_steps_epoch:04d} | SpatialLossScale {spatial_loss_scale:.2f} | "
              f"Loss {running['loss']/max(1,n):.5f} | MSE {running['mse']/max(1,n):.5f} | "
              f"Shape {running['shape']/max(1,n):.5f} | SpatialCorr {running['spatial_corr']/max(1,n):.5f} | Centroid {running['centroid']/max(1,n):.5f} | "
              f"SpatialSpectral {running['spatial_spectral']/max(1,n):.5f}{nf_note}")
    return model, history


def finetune_rollout_consistency(model, loader_rollout, epochs=EPOCHS_ROLLOUT_FINETUNE,
                                  k_steps=K_ROLLOUT_STEPS, k_steps_start=K_ROLLOUT_STEPS_START,
                                  chunk_size=TBPTT_CHUNK, probe_dataset=None, probe_every=PROBE_EVERY_EPOCHS,
                                  val_ds_for_checkpoint=None, patience=ROLLOUT_PATIENCE):
    """v7.15 -- new: early stopping + best checkpoint. Every `probe_every`
    epochs the (already existing, v6.3) stability probe is run AND (if it
    doesn't diverge) a QUICK short-rollout MAPE evaluation on `val_ds`
    (`quick_rollout_val_mape`) -- the model state is saved when that MAPE
    improves, and training stops if it doesn't improve for `patience`
    consecutive checks OR if the probe detects divergence. This allows
    raising `EPOCHS_ROLLOUT_FINETUNE` (more epochs over the same ~999
    high-fidelity steps is cheap) without risking ending on an unstable or
    overfit epoch."""
    params = list(model.dynamics_mean.parameters()) + list(model.decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=LR_ROLLOUT)
    key_names = ["loss", "mse", "shape", "spectral", "var_floor", "var_ceil", "growth", "gradient",
                 "spatial_var", "spatial_spectral", "spatial_corr", "interlayer", "anisotropy",
                 "curvature", "diffusion", "centroid"]
    history = {("loss_total" if k == "loss" else f"loss_{k}"): [] for k in key_names}
    history.update({"probe_divergence_step": [], "k_steps_epoch": [], "val_quick_mape": []})

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    lr_backoffs = 0   # v7.19 -- counter of consecutive LR reductions (epoch rollback)

    for epoch in range(epochs):
        frac = epoch / max(1, epochs - 1)
        k_steps_epoch = int(round(k_steps_start + frac * (k_steps - k_steps_start)))
        k_steps_epoch = max(chunk_size, min(k_steps, k_steps_epoch))
        # v7.17 -- new: warm-up ramp for W_SPATIAL_SPECTRAL/W_CENTROID (see CHANGELOG.md v7.17)
        spatial_loss_scale = min(1.0, (epoch + 1) / max(1, SPATIAL_LOSS_WARMUP_EPOCHS))

        # v7.19 -- snapshot at the start of the epoch (rollback if it turns out corrupt)
        epoch_snapshot = copy.deepcopy(model.state_dict())
        model.train()
        running = {k: 0.0 for k in key_names}
        n = 0
        n_nonfinite = 0
        shape_by_chunk_sum = None
        for x, y, f, tr in loader_rollout:
            if n >= MAX_ROLLOUT_BATCHES_PER_EPOCH:   # v7.19 -- hard cap here too (rarely reached with 10% high fidelity, but guarantees a bounded epoch)
                break
            x, y, f, tr = x.to(device), y.to(device), f.to(device), tr.to(device)
            tr_epoch = tr[:, :k_steps_epoch]
            totals, shape_chunks, had_nonfinite = train_step_rollout_consistency(
                model, x, y, f, optimizer, k_steps=k_steps_epoch, chunk_size=chunk_size, trend_seq=tr_epoch,
                spatial_loss_scale=spatial_loss_scale)
            if had_nonfinite:
                n_nonfinite += 1
            for k in key_names:
                running[k] += totals[k]
            n += 1
            if shape_by_chunk_sum is None:
                shape_by_chunk_sum = list(shape_chunks)
            else:
                for i, v in enumerate(shape_chunks):
                    if i < len(shape_by_chunk_sum):
                        shape_by_chunk_sum[i] += v
                    else:
                        shape_by_chunk_sum.append(v)

        # v7.19 -- epoch-level rollback + LR backoff (see CHANGELOG.md v7.19)
        nonfinite_frac = n_nonfinite / max(1, n)
        if nonfinite_frac >= NONFINITE_EPOCH_FRACTION:
            lr_backoffs += 1
            model.load_state_dict(epoch_snapshot)
            for g in optimizer.param_groups:
                g["lr"] = g["lr"] * 0.5
            print(f"[Phase 2.5b] Epoch {epoch+1:03d}: {n_nonfinite}/{n} batches with non-finite loss "
                  f"(>= {NONFINITE_EPOCH_FRACTION:.0%}) -- epoch DISCARDED (rollback to snapshot), "
                  f"LR reduced to {optimizer.param_groups[0]['lr']:.2e} "
                  f"(backoff {lr_backoffs}/{MAX_LR_BACKOFFS}).")
            if lr_backoffs >= MAX_LR_BACKOFFS:
                print("[Phase 2.5b] LR backoff limit reached -- stopping this phase; "
                      "the best saved checkpoint (if any) will be restored.")
                break
            continue
        lr_backoffs = 0   # healthy epoch -> reset the consecutive-backoff counter

        for k in key_names:
            hist_key = "loss_total" if k == "loss" else f"loss_{k}"
            history[hist_key].append(running[k] / max(1, n))
        history["k_steps_epoch"].append(k_steps_epoch)

        nf_note = f" | [!] {n_nonfinite}/{n} batches discarded (non-finite)" if n_nonfinite else ""
        print(f"[Phase 2.5b - rollout fine-tuning, high fidelity] Epoch {epoch+1:03d}/{epochs} | k_steps {k_steps_epoch:04d} | "
              f"SpatialLossScale {spatial_loss_scale:.2f} | "
              f"Loss {running['loss']/max(1,n):.5f} | MSE {running['mse']/max(1,n):.5f} | "
              f"Shape {running['shape']/max(1,n):.5f} | Spectral {running['spectral']/max(1,n):.5f} | "
              f"VarFloor {running['var_floor']/max(1,n):.5f} | VarCeil {running['var_ceil']/max(1,n):.5f} | Growth {running['growth']/max(1,n):.5f} | "
              f"Gradient {running['gradient']/max(1,n):.5f} | SpatialVar {running['spatial_var']/max(1,n):.5f} | "
              f"SpatialCorr {running['spatial_corr']/max(1,n):.5f} | Interlayer {running['interlayer']/max(1,n):.5f} | "
              f"Anisotropy {running['anisotropy']/max(1,n):.5f} | Curvature {running['curvature']/max(1,n):.5f} | "
              f"Diffusion {running['diffusion']/max(1,n):.5f} | Centroid {running['centroid']/max(1,n):.5f}{nf_note}")
        if shape_by_chunk_sum is not None and len(shape_by_chunk_sum) > 1:
            shape_by_chunk_avg = [v / max(1, n) for v in shape_by_chunk_sum]
            chunk_str = " ".join(f"{v:.3f}" for v in shape_by_chunk_avg)
            print(f"    Shape by chunk position (0=closest to real context, "
                  f"{len(shape_by_chunk_avg)-1}=farthest): [{chunk_str}]")

        if probe_dataset is not None and ((epoch + 1) % probe_every == 0 or epoch == epochs - 1):
            div_step = probe_long_rollout_stability(model, probe_dataset)
            history["probe_divergence_step"].append(div_step)
            if div_step is not None:
                print("    [!] Divergence detected in the probe -- discarding this epoch for the checkpoint.")
                epochs_no_improve += 1
            elif val_ds_for_checkpoint is not None and len(val_ds_for_checkpoint) > 0:
                quick_mape = quick_rollout_val_mape(model, val_ds_for_checkpoint)
                history["val_quick_mape"].append(quick_mape)
                print(f"    Val MAPE (rollout_mean_only, horizon={CHECKPOINT_EVAL_HORIZON}): {quick_mape:.4f}%")
                if quick_mape < best_val - 1e-3:
                    best_val = quick_mape
                    best_state = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    print("    <- best Phase 2.5 checkpoint so far, saved")
                else:
                    epochs_no_improve += 1

        if val_ds_for_checkpoint is not None and epochs_no_improve >= patience:
            print(f"    [early stopping] no improvement in {patience} consecutive checks -- "
                  f"stopping Phase 2.5 at epoch {epoch+1}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Phase 2.5: restored the best checkpoint (quick Val MAPE = {best_val:.4f}%).")
    return model, history


# ==============================================================================
# 11. PHASE 3 (OPTIONAL) -- PHYSICS FINE-TUNING, no more DDIM tricks
# ==============================================================================
def calibrate_physics_on_ground_truth(model, loader_rollout, epochs=EPOCHS_PHYSICS_CALIBRATION,
                                       lr=LR_PHYSICS_CALIBRATION):
    optimizer = torch.optim.Adam(model.physics.parameters(), lr=lr)
    for epoch in range(epochs):
        total, n = 0.0, 0
        for _, y, _, _ in loader_rollout:
            y = y.to(device)
            optimizer.zero_grad()
            loss = pinn_residual_loss(y, model.physics, dt=DT_PINN)
            loss.backward()
            optimizer.step()
            total += loss.item(); n += 1
        print(f"[Physics calibration - real data] Epoch {epoch+1:03d}/{epochs} | Residual {total/max(1,n):.6f}")
    print("Physics coefficients calibrated on ground truth:")
    model.physics.report()
    return model


def train_step_physics_finetune(model, x_seq, y_seq, fid, optimizer, k_steps,
                                 chunk_size=TBPTT_CHUNK, dt_phys=DT_PINN, w_pinn=W_PINN, trend_seq=None):
    model.train()
    with torch.no_grad():
        z_buffer = model.encoder.encode_sequence(x_seq, fid)
    x_prev = x_seq[:, -1]

    n_chunks = math.ceil(k_steps / chunk_size)
    tot_data, tot_phys = 0.0, 0.0
    step_offset = 0
    for _ in range(n_chunks):
        this_chunk = min(chunk_size, k_steps - step_offset)
        optimizer.zero_grad()
        trend_chunk = trend_seq[:, step_offset:step_offset + this_chunk] if trend_seq is not None else None
        preds_decoded, z_buffer, x_prev = _rollout_chunk_mean_only(
            model, z_buffer, fid, this_chunk, trend_chunk=trend_chunk, x_prev_init=x_prev)
        y_chunk = y_seq[:, step_offset:step_offset + this_chunk]

        loss_data = masked_mse(preds_decoded, y_chunk)
        loss_phys = pinn_residual_loss(preds_decoded, model.physics, dt=dt_phys)
        loss = loss_data + w_pinn * loss_phys
        if not torch.isfinite(loss):   # v7.22: same guard as the other phases
            optimizer.zero_grad()
            break
        loss.backward()
        params = list(model.dynamics_mean.parameters()) + list(model.decoder.parameters()) + list(model.physics.parameters())
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        optimizer.step()

        tot_data += loss_data.item(); tot_phys += loss_phys.item()
        z_buffer = z_buffer.detach()
        x_prev = x_prev.detach()
        step_offset += this_chunk

    return tot_data / n_chunks, tot_phys / n_chunks


def finetune_physics(model, loader_rollout, epochs=EPOCHS_PINN_FINETUNE, k_steps=K_ROLLOUT_STEPS,
                      chunk_size=TBPTT_CHUNK, warmup_epochs=PINN_WARMUP_EPOCHS,
                      freeze_physics=FREEZE_PHYSICS_AFTER_CALIBRATION):
    params = list(model.dynamics_mean.parameters()) + list(model.decoder.parameters())
    if not freeze_physics:
        params += list(model.physics.parameters())
    optimizer = torch.optim.Adam(params, lr=LR_PINN_FINETUNE)
    history = {"loss_data": [], "loss_phys": [], "w_pinn": []}
    for epoch in range(epochs):
        w_pinn_epoch = W_PINN * min(1.0, (epoch + 1) / max(1, warmup_epochs))
        model.train()
        tot_d, tot_p, n = 0.0, 0.0, 0
        for x, y, f, tr in loader_rollout:
            x, y, f, tr = x.to(device), y.to(device), f.to(device), tr.to(device)
            ld, lp = train_step_physics_finetune(model, x, y, f, optimizer, k_steps=k_steps,
                                                  chunk_size=chunk_size, w_pinn=w_pinn_epoch, trend_seq=tr)
            tot_d += ld; tot_p += lp; n += 1
        history["loss_data"].append(tot_d / max(1, n))
        history["loss_phys"].append(tot_p / max(1, n))
        history["w_pinn"].append(w_pinn_epoch)
        print(f"[Phase 3 - PINN] Epoch {epoch+1:03d}/{epochs} | w_pinn {w_pinn_epoch:.4f} | "
              f"Data {tot_d/max(1,n):.5f} | Physics {tot_p/max(1,n):.6f}")
    return model, history


# ==============================================================================
# 12. EVALUATION: FREE ROLLOUT + UNCERTAINTY
# ==============================================================================
@torch.no_grad()
def evaluate_free_rollout(model, dataset, horizon, n_samples_eval=50, use_physics_guidance=False,
                           guidance_scale=2.0, rollout_fn=None):
    model.eval()
    rollout_fn = rollout_fn if rollout_fn is not None else model.rollout
    all_errors = []
    for idx in range(min(n_samples_eval, len(dataset))):
        x_seq, y_seq = dataset[idx]
        x_seq = x_seq.unsqueeze(0).to(device)
        y_seq = y_seq[:horizon].to(device)
        fid = model._default_fidelity(1, device)
        trend_seq = None
        if hasattr(dataset, "get_trend_window"):
            trend_seq = dataset.get_trend_window(idx)[:horizon].unsqueeze(0).to(device)
        if getattr(rollout_fn, "__name__", "") == "rollout_mean_only":
            preds = rollout_fn(x_seq, horizon, fidelity_id=fid, trend_seq=trend_seq)
        else:
            preds = rollout_fn(x_seq, horizon, fidelity_id=fid, trend_seq=trend_seq,
                                use_physics_guidance=use_physics_guidance, guidance_scale=guidance_scale)
        preds = denormalize_tensor(preds.squeeze(0), v_min, v_max)
        y_seq = denormalize_tensor(y_seq, v_min, v_max)
        step_errors = []
        for h in range(horizon):
            mask = y_seq[h] != 0.0
            if mask.sum() == 0:
                step_errors.append(0.0)
                continue
            mape = (torch.abs((y_seq[h][mask] - preds[h][mask]) / y_seq[h][mask]).mean() * 100).item()
            step_errors.append(mape)
        all_errors.append(step_errors)
    return np.array(all_errors)


@torch.no_grad()
def quick_rollout_val_mape(model, val_ds, horizon=CHECKPOINT_EVAL_HORIZON, n_samples=10):
    """v7.15 -- Cheap metric for early-stopping/checkpointing in Phase 2.5.
    Deliberately cheap (few samples) so it can be called every
    `probe_every` epochs without doubling training cost -- it does not
    replace `evaluate_free_rollout` (more exhaustive, used in final
    evaluation).

    v7.17 -- default horizon raised from 15 to `CHECKPOINT_EVAL_HORIZON`
    (200): with horizon=15, this metric was BLIND to the long-horizon
    improvement Phase 2.5 exists to give -- a real run showed this made
    early stopping ALWAYS prefer the earliest epoch (least long-horizon
    training), not because it was actually best, but because the selection
    metric couldn't see what was being gained/lost beyond 15 steps. See
    CHANGELOG.md v7.17."""
    if val_ds is None or len(val_ds) == 0:
        return float("inf")
    horizon = min(horizon, getattr(val_ds, "horizon_len", horizon))
    errors = evaluate_free_rollout(model, val_ds, horizon, n_samples_eval=n_samples,
                                    rollout_fn=model.rollout_mean_only)
    return float(errors.mean()) if errors.size > 0 else float("inf")


@torch.no_grad()
def check_flow_diversity(model, x_seq, n_samples=16, fidelity_id=None):
    model.eval()
    if fidelity_id is None:
        fidelity_id = model._default_fidelity(x_seq.shape[0], x_seq.device)
    z_buffer = model.encode_window(x_seq, fidelity_id)
    z_mean = model.dynamics_mean(z_buffer)
    cond = torch.cat([z_mean, z_buffer[:, -1]], dim=-1)
    samples = torch.stack(
        [model.dynamics_flow.sample(cond, n_steps=N_FLOW_STEPS, stochastic=False) for _ in range(n_samples)],
        dim=0,
    )
    std_across_samples = samples.std(dim=0).mean().item()
    scale = z_mean.abs().mean().item() + 1e-8
    return {
        "std_across_samples": std_across_samples,
        "z_mean_scale": scale,
        "diversity_ratio": std_across_samples / scale,
    }


def print_flow_diversity_report(report):
    print("\n=== DIAGNOSTIC: FLOW-MATCHING RESIDUAL DIVERSITY (latent space) ===")
    print(f"  std across samples (same context, different z0): {report['std_across_samples']:.6f}")
    print(f"  typical scale of |z_mean|:                        {report['z_mean_scale']:.6f}")
    print(f"  diversity_ratio (std / scale):                    {report['diversity_ratio']:.6f}")
    ratio = report["diversity_ratio"]
    if ratio < 0.02:
        print("  [!] Very low ratio -- the flow appears to have almost completely collapsed to "
              "ignoring z0 (practically deterministic regardless of input noise).")
    elif ratio < 0.15:
        print("  [!] Low ratio -- the flow has likely PARTIALLY collapsed.")
    else:
        print("  Healthy ratio -- the flow DOES respond to input noise with reasonable diversity.")


@torch.no_grad()
def estimate_predictive_uncertainty(model, x_seq, n_steps, n_samples=8, guidance_scale=0.0):
    model.eval()
    fid = model._default_fidelity(x_seq.shape[0], x_seq.device)
    samples = []
    for _ in range(n_samples):
        preds = model.rollout(x_seq, n_steps, fidelity_id=fid, stochastic=True, guidance_scale=guidance_scale)
        samples.append(preds.unsqueeze(0))
    samples = torch.cat(samples, dim=0)
    return samples.mean(dim=0), samples.std(dim=0)


def plot_rollout_error(errors, save_dir=SAVE_DIR, title="Error degradation in free rollout (v7.33)"):
    os.makedirs(save_dir, exist_ok=True)
    mean_err = errors.mean(axis=0)
    plt.figure(figsize=(9, 5))
    plt.plot(np.arange(1, len(mean_err) + 1), mean_err, marker="o", linewidth=2)
    plt.xlabel("Future step (t+n)")
    plt.ylabel("MAPE (%)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(save_dir, "rollout_error_v6.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ==============================================================================
# 13. EXECUTION -- 4 PHASES (1, 2, 2.5 new, 3 optional)
# ==============================================================================
if __name__ == "__main__":
    loaders = make_loaders()
    model = LatentWorldModelV6(LATENT_DIM).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model v7.33 created. Parameters: {n_params:,}")

    print("\n=== PHASE 1a: autoencoder pretraining (low + medium) ===")
    model, hist_ae_pretrain = pretrain_autoencoder(model, loaders["loader_low_ae"], loaders["loader_medium_ae"])

    print("\n=== PHASE 1b: autoencoder fine-tuning (high fidelity) ===")
    model, hist_ae_finetune = finetune_autoencoder(model, loaders["loader_high_ae"], val_loader=loaders["val_loader_ae"])

    print("\n=== PHASE 2a: dynamics pretraining (transformer + flow matching) ===")
    model, hist_dyn_pretrain = pretrain_dynamics(model, loaders["loader_low_dyn"], loaders["loader_medium_dyn"])

    print("\n=== PHASE 2b: dynamics fine-tuning (high fidelity) ===")
    model, hist_dyn_finetune = finetune_dynamics(model, loaders["loader_high_dyn"])

    hist_rollout = None
    hist_rollout_pretrain = None
    if RUN_PHASE_ROLLOUT:
        print("\n=== PHASE 2.5a (NEW v7.18): rollout-consistency pretraining (low+medium) ===")
        model, hist_rollout_pretrain = pretrain_rollout_consistency(
            model, loaders["loader_rollout_low"], loaders["loader_rollout_medium"])

        print("\n=== PHASE 2.5b: rollout-consistency fine-tuning (high fidelity) -- "
              "exposure-bias fix ===")
        model, hist_rollout = finetune_rollout_consistency(
            model, loaders["loader_rollout_high"],
            probe_dataset=loaders["probe_ds"], val_ds_for_checkpoint=loaders["val_ds"])
    else:
        print("\n[RUN_PHASE_ROLLOUT=False] Skipped -- WARNING: without this, free rollout will "
              "likely keep diverging just like in v5.")

    hist_pinn = None
    if RUN_PHASE3_PINN:
        if RUN_PHYSICS_CALIBRATION:
            print("\n=== PHYSICS CALIBRATION: v_z/alpha/S on ground truth ===")
            model = calibrate_physics_on_ground_truth(model, loaders["loader_rollout_high"])
        # v7.22 -- snapshot + automatic reversion (see CHANGELOG.md v7.22):
        # Phase 3 can no longer leave the model worse than it found it.
        pre_pinn_state = copy.deepcopy(model.state_dict())
        pre_pinn_mape = quick_rollout_val_mape(model, loaders["val_ds"])
        print(f"\n[Phase 3] Reference Val MAPE BEFORE physics tuning: {pre_pinn_mape:.4f}%")
        print("\n=== PHASE 3: physics fine-tuning (PINN), natively differentiable ===")
        model, hist_pinn = finetune_physics(model, loaders["loader_rollout_high"])
        model.physics.report()
        post_pinn_mape = quick_rollout_val_mape(model, loaders["val_ds"])
        post_div = probe_long_rollout_stability(model, loaders["probe_ds"])
        rel_change = (post_pinn_mape - pre_pinn_mape) / max(pre_pinn_mape, 1e-8)
        print(f"[Phase 3] Val MAPE AFTER: {post_pinn_mape:.4f}% (relative change {rel_change:+.1%})")
        if post_div is not None or rel_change > PINN_ACCEPT_TOLERANCE:
            model.load_state_dict(pre_pinn_state)
            print(f"[Phase 3] RESULT REJECTED ({'probe diverged' if post_div is not None else 'Val MAPE worsened beyond ' + format(PINN_ACCEPT_TOLERANCE, '.0%')}) "
                  f"-- model restored to its pre-Phase-3 state. Physics tuning discarded for this run.")
        else:
            print("[Phase 3] Result ACCEPTED (stable and Val MAPE not worsened beyond tolerance).")
    else:
        print("\n[RUN_PHASE3_PINN=False] Physics tuning skipped.")

    val_ds = loaders["val_ds"]
    test_ds = loaders["test_ds"]
    test_loader = loaders["test_loader"]

    print("\n=== FINAL EVALUATION: free rollout on high-fidelity TEST ===")
    if len(test_ds) > 0:
        horizon = test_ds.horizon_len
        print("-- model.rollout (mean + flow residual) --")
        errors = evaluate_free_rollout(model, test_ds, horizon)
        mean_err = errors.mean(axis=0)
        for i, e in enumerate(mean_err):
            print(f"  t+{i+1:03d}: {e:.4f}%  (std across samples: {errors.std(axis=0)[i]:.4f}%)")
        plot_rollout_error(errors)

        print("\n-- model.rollout_mean_only (Transformer only, no flow) --")
        errors_mean_only = evaluate_free_rollout(model, test_ds, horizon, rollout_fn=model.rollout_mean_only)
        mean_err_only = errors_mean_only.mean(axis=0)
        for i, e in enumerate(mean_err_only):
            print(f"  t+{i+1:03d}: {e:.4f}%  (std across samples: {errors_mean_only.std(axis=0)[i]:.4f}%)")

        x_seq_eval, _ = test_ds[0]
        x_seq_eval = x_seq_eval.unsqueeze(0).to(device)
        mean_pred, std_pred = estimate_predictive_uncertainty(model, x_seq_eval, n_steps=min(8, horizon))
        _scale_shape = [1] * std_pred.dim()
        _scale_shape[-3] = 9
        _scale = torch.as_tensor(v_max - v_min, dtype=std_pred.dtype, device=std_pred.device).reshape(_scale_shape)
        std_real = std_pred * _scale
        print("\n=== PREDICTIVE UNCERTAINTY (std across 8 stochastic rollouts) ===")
        per_step = std_real.mean(dim=(0, 2, 3, 4))
        for i, s in enumerate(per_step[:5]):
            print(f"  t+{i+1:03d}: std ~ {s.item():.4f}")

        flow_report = check_flow_diversity(model, x_seq_eval)
        print_flow_diversity_report(flow_report)

        print("\n=== FINAL STABILITY PROBE (long horizon, TEST) ===")
        probe_horizon_test = min(PROBE_HORIZON, max(1, len(data_test) - SEQ_LEN))
        probe_ds_test = ReactorWindowDataset(data_test, horizon_len=probe_horizon_test,
                                              trend_override=DMD_FORECAST_TEST)
        probe_long_rollout_stability(model, probe_ds_test)
    else:
        print("[!] Not enough high-fidelity test data for a full rollout.")

    # ------------------------------------------------------------------
    # Checkpoint (this delivery, not in the original v7.33 notebook cells):
    # the diagnostics script was originally meant to be pasted into the SAME
    # kernel session right after this training script finished, so it could
    # just read `model`, `test_ds`, `v_min`, etc. directly out of that
    # session's memory. Run from a terminal instead (`python3
    # reactor_world_model.py` then, later, `python3
    # reactor_world_model_diagnostics.py` as a SEPARATE process), those
    # variables no longer exist -- hence saving them here, so the diagnostics
    # script can load this file and rebuild everything it needs on its own.
    # See CHANGELOG.md and README.md ("How to run") for the full explanation.
    os.makedirs("outputs", exist_ok=True)
    checkpoint_path = os.path.join("outputs", "checkpoint_v7_33.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "v_min": v_min,
        "v_max": v_max,
        "hist_ae_pretrain": hist_ae_pretrain,
        "hist_ae_finetune": hist_ae_finetune,
        "hist_dyn_pretrain": hist_dyn_pretrain,
        "hist_dyn_finetune": hist_dyn_finetune,
        "hist_rollout": hist_rollout,
        "hist_pinn": hist_pinn,
    }, checkpoint_path)
    print(f"\nCheckpoint saved to '{checkpoint_path}'. You can now run "
          f"reactor_world_model_diagnostics.py as a separate command -- it will "
          f"detect and load this checkpoint automatically.")