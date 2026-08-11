"""
===============================================================================
DIAGNOSTIC PLOTS FOR `LatentWorldModelV6` -- v7.33
===============================================================================
Paste this into a NEW cell after running `reactor_world_model.py` (Phases 1,
2, 2.5, and optionally 3). It reuses variables left in the kernel: model,
test_ds, val_ds, test_loader, device, v_min, v_max, denormalize_tensor,
masked_mape, masked_spatial_mean_series, SEQ_LEN, N_FLOW_STEPS, DT_PINN,
RUN_PHASE_ROLLOUT, RUN_PHASE3_PINN, make_physics_guidance, hist_ae_pretrain,
hist_ae_finetune, hist_dyn_pretrain, hist_dyn_finetune, hist_rollout_pretrain,
hist_rollout, hist_pinn.

The full version-by-version development history of this project (v5 ->
v7.33) lives in CHANGELOG.md at the repository root -- comments below point
back to it by version tag instead of duplicating that narrative.

WHAT CHANGED TO REACH v7.33 (updated from the originally supplied v7.25
diagnostics baseline)
-------------------------------------------------------------------------------
`create_rollout_video` (and its call in the execution section) has been
updated to match the specification from the v7.33 model-script addendum
("first 100 steps, all 9 layers, one frame per step, none skipped"): the
function's defaults changed from `n_steps=200, layers=(0, 4, 8),
frame_stride=2` to `n_steps=100, layers=tuple(range(9)), frame_stride=1`,
and the call at the bottom of this script now requests exactly that. This
was the one place where the originally delivered diagnostics file (tagged
v7_25) had fallen behind what the v7.33 model script already promised.
Everything else in this file carries forward unchanged in behavior, only
translated to English.

Earlier milestones (all behavior-preserving, see CHANGELOG.md for full
detail): v7.25 fixed two bugs found while using the v7.24 tools on real
data (`estimate_dominant_period` locking onto the search floor under a
slowly-drifting mean; `plot_concentration_flow_field`'s divergence maps
coming out mostly NaN from a scale-dependent absolute threshold). v7.24
added the concentration flow-field diagnostic (`estimate_flow_field_lk`,
`plot_concentration_flow_field`) implementing the user's vector-field idea.
v7.20 made the continuous simulation also apply the new per-pixel texture
envelope. v7.18 added `plot_density_peaks_check` and
`plot_full_cycle_replication_check`. v7.17 changed
`check_bifurcation_is_static`'s default layer to 0 and added
`plot_turbulence_statistics_check`. v7.15 added
`plot_texture_directionality_check`. Everything else (error matrix, spatial
maps, latent-space diagnostics, uncertainty, ablations,
`check_bifurcation_is_static`, `plot_centroid_tracking`,
`plot_transition_trigger_analysis`, `report_extreme_mape_pixels`) has kept
the same API since v7.14 -- the model's API (rollout, rollout_mean_only,
encode_window, decoder, apply_*) has not changed.
===============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

SAVE_DIR = "outputs/plots_v6"
os.makedirs(SAVE_DIR, exist_ok=True)


# ==============================================================================
# 1. ERROR MATRIX (horizon x 9 layers) -- full-quality rollout
# ==============================================================================
@torch.no_grad()
def evaluate_error_matrix(model, loader, n_steps, device, v_min, v_max):
    model.eval()
    mape_matrix_accum = np.zeros((n_steps, 9))
    total_batches = 0
    has_trend = hasattr(loader.dataset, "get_trend_window")
    offset = 0

    for x_seq, y_seq in loader:
        batch_size = x_seq.shape[0]
        x_seq, y_seq = x_seq.to(device), y_seq.to(device)
        y_seq = y_seq[:, :n_steps]

        trend_seq = None
        if has_trend:
            trend_seq = torch.stack([
                loader.dataset.get_trend_window(offset + i)[:n_steps] for i in range(batch_size)
            ]).to(device)
        preds = model.rollout(x_seq, n_steps, trend_seq=trend_seq)
        preds_real = denormalize_tensor(preds, v_min, v_max)
        target_real = denormalize_tensor(y_seq, v_min, v_max)

        for h in range(n_steps):
            for layer in range(9):
                mape_matrix_accum[h, layer] += masked_mape(
                    preds_real[:, h, layer], target_real[:, h, layer]
                ).item()
        total_batches += 1
        offset += batch_size

    return mape_matrix_accum / max(1, total_batches)


# ==============================================================================
# 2. TRAINING HISTORY -- 4 phases
# ==============================================================================
def plot_training_overview(hist_ae_pretrain, hist_ae_finetune, hist_dyn_pretrain, hist_dyn_finetune,
                            hist_rollout=None, hist_pinn=None, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(19, 9))

    ae_loss = hist_ae_pretrain["train_loss"] + hist_ae_finetune["train_loss"]
    boundary_ae = len(hist_ae_pretrain["train_loss"])
    ep_ae = np.arange(1, len(ae_loss) + 1)
    axes[0, 0].plot(ep_ae, ae_loss, color="steelblue")
    if 0 < boundary_ae < len(ep_ae):
        axes[0, 0].axvline(ep_ae[boundary_ae], color="gray", linestyle="--", alpha=0.8, label="fine-tuning starts")
        axes[0, 0].legend()
    axes[0, 0].set_title("Phase 1 -- Autoencoder loss\n(reconstruction, pretrain -> fine-tuning)")
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")

    val_mape = hist_ae_finetune.get("val_mape", [])
    if len(val_mape) > 0:
        ep_val = np.arange(boundary_ae + 1, boundary_ae + 1 + len(val_mape))
        axes[0, 1].plot(ep_val, val_mape, color="darkorange", marker="o")
        axes[0, 1].set_title("Phase 1 -- Reconstruction val MAPE\n(fine-tuning only, high fidelity)")
        axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("MAPE (%)")
    else:
        axes[0, 1].text(0.5, 0.5, "No val_loader passed to\nfinetune_autoencoder", ha="center", va="center")
        axes[0, 1].axis("off")

    dyn_mean = hist_dyn_pretrain["mse_mean"] + hist_dyn_finetune["mse_mean"]
    dyn_flow = hist_dyn_pretrain["flow_loss"] + hist_dyn_finetune["flow_loss"]
    boundary_dyn = len(hist_dyn_pretrain["mse_mean"])
    ep_dyn = np.arange(1, len(dyn_mean) + 1)
    axes[0, 2].plot(ep_dyn, dyn_mean, color="seagreen", label="Transformer mean (MSE)")
    axes[0, 2].plot(ep_dyn, dyn_flow, color="darkviolet", label="Flow matching loss")
    if 0 < boundary_dyn < len(ep_dyn):
        axes[0, 2].axvline(ep_dyn[boundary_dyn], color="gray", linestyle="--", alpha=0.8, label="fine-tuning starts")
    axes[0, 2].set_title("Phase 2 -- One-step dynamics loss\n(Transformer + rectified flow)")
    axes[0, 2].set_xlabel("Epoch"); axes[0, 2].set_ylabel("Loss"); axes[0, 2].legend()

    if hist_rollout is not None and len(hist_rollout.get("loss_total", [])) > 0:
        ep_ro = np.arange(1, len(hist_rollout["loss_total"]) + 1)
        axes[1, 0].plot(ep_ro, hist_rollout["loss_total"], color="black", linewidth=2, label="Total")
        axes[1, 0].plot(ep_ro, hist_rollout["loss_mse"], color="crimson", linestyle="--", label="Masked MSE")
        axes[1, 0].plot(ep_ro, hist_rollout["loss_shape"], color="teal", linestyle="--", label="Shape (1-corr)")
        if "loss_spectral" in hist_rollout:
            axes[1, 0].plot(ep_ro, hist_rollout["loss_spectral"], color="darkorange", linestyle=":", label="Spectral temporal (v6.1)")
        if "loss_var_floor" in hist_rollout:
            axes[1, 0].plot(ep_ro, hist_rollout["loss_var_floor"], color="purple", linestyle=":", label="Variance floor (v6.1)")
        if "loss_var_ceil" in hist_rollout:
            axes[1, 0].plot(ep_ro, hist_rollout["loss_var_ceil"], color="slateblue", linestyle=":", label="Variance ceiling (v7.21, NEW)")
        if "loss_growth" in hist_rollout:
            axes[1, 0].plot(ep_ro, hist_rollout["loss_growth"], color="firebrick", linestyle="-.", label="Growth penalty (v6.3)")
        if "loss_gradient" in hist_rollout:
            axes[1, 0].plot(ep_ro, hist_rollout["loss_gradient"], color="darkgreen", linestyle="--", label="Spatial gradient (v7.7, w=0 default)")
        if "loss_spatial_var" in hist_rollout:
            axes[1, 0].plot(ep_ro, hist_rollout["loss_spatial_var"], color="magenta", linestyle=":", label="Spatial variance (v7.7, w=0 default)")
        if "loss_spatial_corr" in hist_rollout:
            axes[1, 0].plot(ep_ro, hist_rollout["loss_spatial_corr"], color="darkcyan", linestyle=":", label="Spatial pattern corr (v7.22, NEW)")
        if "loss_spatial_spectral" in hist_rollout:
            axes[1, 0].plot(ep_ro, hist_rollout["loss_spatial_spectral"], color="blue", linestyle=":", label="Spatial spectral (v7.15, NEW)")
        if "loss_centroid" in hist_rollout:
            axes[1, 0].plot(ep_ro, hist_rollout["loss_centroid"], color="brown", linestyle=":", label="Centroid/spread (v7.15, NEW)")
        axes[1, 0].set_title("Phase 2.5 -- Rollout-consistency loss\n(exposure-bias / collapse / stability / texture / centroid fixes)")
        axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Loss"); axes[1, 0].legend(fontsize=7)
        divergence_steps = hist_rollout.get("probe_divergence_step", [])
        if len(divergence_steps) > 0:
            last = divergence_steps[-1]
            msg = f"last stability probe: diverged at t+{last+1}" if last is not None else "last stability probe: no divergence"
            axes[1, 0].text(0.02, 0.02, msg, transform=axes[1, 0].transAxes, fontsize=7,
                             color=("crimson" if last is not None else "green"), va="bottom")
    else:
        axes[1, 0].text(0.5, 0.5, "RUN_PHASE_ROLLOUT=False\n(not run)", ha="center", va="center")
        axes[1, 0].axis("off")

    if hist_pinn is not None and len(hist_pinn.get("loss_data", [])) > 0:
        ep_pinn = np.arange(1, len(hist_pinn["loss_data"]) + 1)
        ax_l = axes[1, 1]
        ax_r = ax_l.twinx()
        ax_l.plot(ep_pinn, hist_pinn["loss_data"], color="steelblue", marker="o", label="Data loss")
        ax_r.plot(ep_pinn, hist_pinn["loss_phys"], color="crimson", marker="s", label="Physics residual")
        ax_l.set_xlabel("Epoch"); ax_l.set_ylabel("Data loss", color="steelblue")
        ax_r.set_ylabel("Physics residual", color="crimson")
        ax_l.set_title("Phase 3 -- PINN fine-tuning")
        lines1, labels1 = ax_l.get_legend_handles_labels()
        lines2, labels2 = ax_r.get_legend_handles_labels()
        ax_l.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        axes[1, 1].text(0.5, 0.5, "RUN_PHASE3_PINN=False\n(not run)", ha="center", va="center")
        axes[1, 1].axis("off")

    if hist_rollout is not None and len(hist_rollout.get("val_quick_mape", [])) > 0:
        vq = hist_rollout["val_quick_mape"]
        axes[1, 2].plot(np.arange(1, len(vq) + 1), vq, color="darkgreen", marker="o")
        axes[1, 2].set_title("Phase 2.5 -- Quick Val MAPE\n(used for early-stopping/checkpoint, v7.15)")
        axes[1, 2].set_xlabel("Check (every probe_every epochs)"); axes[1, 2].set_ylabel("MAPE (%)")
        axes[1, 2].grid(True, linestyle="--", alpha=0.5)
    else:
        axes[1, 2].axis("off")

    plt.tight_layout()
    path = os.path.join(save_dir, "training_history_v6.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


def plot_layer_mape_t1(error_matrix, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    layer_mapes_t1 = error_matrix[0, :]
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(9), layer_mapes_t1, marker="o", color="purple")
    plt.title("MAPE per Axial Layer (Step t+1)")
    plt.xlabel("Layer (0=Base, 8=Top)"); plt.ylabel("MAPE (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(save_dir, "layer_mape_t1.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


# ==============================================================================
# 3. ERROR PROPAGATION OVER THE HORIZON
# ==============================================================================
def plot_error_propagation(error_matrix, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    global_mape_per_step = error_matrix.mean(axis=1)
    steps = np.arange(1, len(global_mape_per_step) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(steps, global_mape_per_step, marker="o", color="#1f77b4", linewidth=2.5)
    plt.title("Error Propagation Over Time (v7.33 rollout, high fidelity)")
    plt.xlabel("Future step (t+n)"); plt.ylabel("Global MAPE (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    path = os.path.join(save_dir, "error_propagation_horizon.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


# ==============================================================================
# 4. SPATIAL ERROR PER LAYER AT t+N
# ==============================================================================
@torch.no_grad()
def plot_spatial_error(model, dataset, sample_idx, device, v_min, v_max, n_steps=6, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)

    x_seq_sample, y_targets_sample = dataset[sample_idx]
    model.eval()

    x_seq = x_seq_sample.unsqueeze(0).to(device)
    trend_seq = dataset.get_trend_window(sample_idx)[:n_steps].unsqueeze(0).to(device) if hasattr(dataset, "get_trend_window") else None
    preds = model.rollout(x_seq, n_steps, trend_seq=trend_seq).squeeze(0).cpu()

    pred_sample_real = denormalize_tensor(preds[n_steps - 1], v_min, v_max).numpy()
    real_sample_real = denormalize_tensor(y_targets_sample[n_steps - 1], v_min, v_max).numpy()

    real_core = np.where(real_sample_real == 0.0, np.nan, real_sample_real)
    pred_core = np.where(np.isnan(real_core), np.nan, pred_sample_real)

    fig, ax = plt.subplots(9, 3, figsize=(12, 28))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for layer in range(9):
            vmin = np.nanmin([real_core[layer], pred_core[layer]])
            vmax = np.nanmax([real_core[layer], pred_core[layer]])

            im0 = ax[layer, 0].imshow(real_core[layer], vmin=vmin, vmax=vmax, cmap="viridis")
            ax[layer, 0].set_title(f"Real L{layer} (t+{n_steps})")
            ax[layer, 0].axis("off")
            fig.colorbar(im0, ax=ax[layer, 0], fraction=0.046, pad=0.04)

            im1 = ax[layer, 1].imshow(pred_core[layer], vmin=vmin, vmax=vmax, cmap="viridis")
            ax[layer, 1].set_title(f"Predicted L{layer} (t+{n_steps})")
            ax[layer, 1].axis("off")
            fig.colorbar(im1, ax=ax[layer, 1], fraction=0.046, pad=0.04)

            mape_map = np.abs((real_core[layer] - pred_core[layer]) / real_core[layer]) * 100
            im2 = ax[layer, 2].imshow(mape_map, cmap="magma")
            ax[layer, 2].set_title(f"MAPE (%) L{layer}")
            ax[layer, 2].axis("off")
            fig.colorbar(im2, ax=ax[layer, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    filename = f"spatial_error_t{n_steps}_sample_{sample_idx}.png"
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


# ==============================================================================
# 4b. MULTI-HORIZON SPATIAL ERROR EVOLUTION
# ==============================================================================
@torch.no_grad()
def plot_mape_evolution_multi_horizon(model, dataset, sample_idx, device, v_min, v_max,
                                       horizons=(10, 50, 100, 200, 500, 750, 1000), save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    max_h = max(horizons)
    available_h = getattr(dataset, "horizon_len", max_h)
    if max_h > available_h:
        print(f"[!] max(horizons)={max_h} > dataset horizon_len ({available_h}) -- "
              f"requested horizons are clipped to what's available.")
        horizons = tuple(h for h in horizons if h <= available_h) or (available_h,)
        max_h = max(horizons)

    x_seq_sample, y_targets_sample = dataset[sample_idx]
    x_seq = x_seq_sample.unsqueeze(0).to(device)
    trend_seq = (dataset.get_trend_window(sample_idx)[:max_h].unsqueeze(0).to(device)
                 if hasattr(dataset, "get_trend_window") else None)
    preds = model.rollout(x_seq, max_h, trend_seq=trend_seq).squeeze(0).cpu()

    mape_grid = np.zeros((9, len(horizons)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for hi, h in enumerate(horizons):
            pred_h = denormalize_tensor(preds[h - 1], v_min, v_max).numpy()
            real_h = denormalize_tensor(y_targets_sample[h - 1], v_min, v_max).numpy()
            real_core = np.where(real_h == 0.0, np.nan, real_h)
            pred_core = np.where(np.isnan(real_core), np.nan, pred_h)
            mape_map = np.abs((real_core - pred_core) / real_core) * 100
            for layer in range(9):
                mape_grid[layer, hi] = np.nanmean(mape_map[layer])

    fig1, ax1 = plt.subplots(figsize=(1.3 * len(horizons) + 3, 6))
    im = ax1.imshow(mape_grid, aspect="auto", cmap="magma")
    ax1.set_xticks(range(len(horizons))); ax1.set_xticklabels([f"t+{h}" for h in horizons])
    ax1.set_yticks(range(9)); ax1.set_yticklabels([f"Layer {l}" for l in range(9)])
    ax1.set_title("Mean MAPE (%) per layer vs. rollout horizon (single free rollout)")
    fig1.colorbar(im, ax=ax1, label="MAPE (%)")
    for layer in range(9):
        for hi in range(len(horizons)):
            ax1.text(hi, layer, f"{mape_grid[layer, hi]:.2f}", ha="center", va="center",
                      color="white" if mape_grid[layer, hi] < np.nanmax(mape_grid) * 0.6 else "black", fontsize=7)
    plt.tight_layout()
    path1 = os.path.join(save_dir, "mape_heatmap_layer_vs_horizon.png")
    plt.savefig(path1, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path1}")

    rep_layers = [0, 4, 8]
    fig2, ax2 = plt.subplots(len(rep_layers) * 2, len(horizons), figsize=(2.6 * len(horizons), 4 * len(rep_layers)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for hi, h in enumerate(horizons):
            pred_h = denormalize_tensor(preds[h - 1], v_min, v_max).numpy()
            real_h = denormalize_tensor(y_targets_sample[h - 1], v_min, v_max).numpy()
            real_core = np.where(real_h == 0.0, np.nan, real_h)
            pred_core = np.where(np.isnan(real_core), np.nan, pred_h)
            for li, layer in enumerate(rep_layers):
                vmin = np.nanmin([real_core[layer], pred_core[layer]])
                vmax = np.nanmax([real_core[layer], pred_core[layer]])
                row_real, row_pred = 2 * li, 2 * li + 1
                ax2[row_real, hi].imshow(real_core[layer], vmin=vmin, vmax=vmax, cmap="viridis")
                ax2[row_real, hi].axis("off")
                ax2[row_pred, hi].imshow(pred_core[layer], vmin=vmin, vmax=vmax, cmap="viridis")
                ax2[row_pred, hi].axis("off")
                if li == 0:
                    ax2[row_real, hi].set_title(f"t+{h}", fontsize=10)
    for li, layer in enumerate(rep_layers):
        ax2[2 * li, 0].set_ylabel(f"Real L{layer}", fontsize=9)
        ax2[2 * li, 0].axis("on"); ax2[2 * li, 0].set_xticks([]); ax2[2 * li, 0].set_yticks([])
        ax2[2 * li + 1, 0].set_ylabel(f"Pred L{layer}", fontsize=9)
        ax2[2 * li + 1, 0].axis("on"); ax2[2 * li + 1, 0].set_xticks([]); ax2[2 * li + 1, 0].set_yticks([])
    plt.suptitle(f"Spatial field evolution across horizons -- layers {rep_layers} "
                 "(base/mid/top, see the compact MAPE heatmap above for all 9 layers)", y=1.01)
    plt.tight_layout()
    path2 = os.path.join(save_dir, "spatial_field_evolution_horizons.png")
    plt.savefig(path2, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path2}")

    return mape_grid


# ==============================================================================
# 4c. TEXTURE DIRECTIONALITY RATIO -- v7.15, NEW
# ==============================================================================
def _spatial_gradient_energy_np(field, mask):
    """field, mask: (H,W). Spatial-gradient energy (average of |horizontal
    diff| + |vertical diff|), only between pairs of cells BOTH inside the
    mask -- a simple, direct measure of how much DIRECTIONAL structure the
    field has. A perfectly uniform blob gives ~0; the real field (with its
    axial/radial gradient) gives a clearly positive value."""
    vals = []
    if mask.shape[1] > 1:
        gh = np.abs(field[:, 1:] - field[:, :-1])
        mh = mask[:, 1:] & mask[:, :-1]
        if mh.any():
            vals.append(gh[mh])
    if mask.shape[0] > 1:
        gv = np.abs(field[1:, :] - field[:-1, :])
        mv = mask[1:, :] & mask[:-1, :]
        if mv.any():
            vals.append(gv[mv])
    if not vals:
        return 0.0
    return float(np.concatenate(vals).mean())


@torch.no_grad()
def plot_texture_directionality_check(model, dataset, device, v_min, v_max,
                                       horizons=(15, 100, 500, 1000), n_samples=10, save_dir=SAVE_DIR):
    """v7.15 -- NEW. Aggregate MAPE cannot distinguish "lost all directional
    texture (blob collapse)" from a genuine amplitude error -- especially in
    small-physical-range layers (L4-L8), where MAPE% can stay low even
    though the prediction is nearly uniform (confirmed in a real run with
    HIGH_FIDELITY_TRAIN_FRAC=0.10: L4-L8 looked flat in spatial images from
    t+50 onward, with equally low MAPE% due to those layers' tiny physical
    range). This function DIRECTLY measures how much spatial structure
    (gradient energy) the prediction has vs. reality, per layer and
    horizon -- a ratio << 1 is the quantitative signature of blob collapse,
    regardless of how low MAPE comes out. Always run this alongside
    `plot_mape_evolution_multi_horizon`."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    max_h = max(horizons)
    ratio_accum = {h: np.zeros(9) for h in horizons}
    count_accum = {h: np.zeros(9) for h in horizons}

    for idx in range(min(n_samples, len(dataset))):
        x_seq, y_seq = dataset[idx]
        this_max_h = min(max_h, y_seq.shape[0])
        if this_max_h < 1:
            continue
        x_seq_b = x_seq.unsqueeze(0).to(device)
        trend_seq = (dataset.get_trend_window(idx)[:this_max_h].unsqueeze(0).to(device)
                     if hasattr(dataset, "get_trend_window") else None)
        preds = model.rollout(x_seq_b, this_max_h, trend_seq=trend_seq).squeeze(0).cpu()
        preds_real = denormalize_tensor(preds, v_min, v_max).numpy()
        real_real = denormalize_tensor(y_seq[:this_max_h], v_min, v_max).numpy()
        for h in horizons:
            if h > this_max_h:
                continue
            for layer in range(9):
                mask = real_real[h - 1, layer] != 0.0
                if mask.sum() < 4:
                    continue
                e_real = _spatial_gradient_energy_np(real_real[h - 1, layer], mask)
                e_pred = _spatial_gradient_energy_np(preds_real[h - 1, layer], mask)
                if e_real > 1e-9:
                    ratio_accum[h][layer] += e_pred / e_real
                    count_accum[h][layer] += 1

    ratio_grid = np.full((9, len(horizons)), np.nan)
    for hi, h in enumerate(horizons):
        counts = count_accum[h]
        for layer in range(9):
            if counts[layer] > 0:
                ratio_grid[layer, hi] = ratio_accum[h][layer] / counts[layer]

    fig, ax = plt.subplots(figsize=(1.5 * len(horizons) + 3, 6))
    im = ax.imshow(ratio_grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=2)
    ax.set_xticks(range(len(horizons))); ax.set_xticklabels([f"t+{h}" for h in horizons])
    ax.set_yticks(range(9)); ax.set_yticklabels([f"Layer {l}" for l in range(9)])
    ax.set_title("Texture directionality ratio (predicted / real spatial-gradient energy)\n"
                 "1.0 = matches real texture; << 1 = collapsing to a flat blob (invisible to MAPE)")
    fig.colorbar(im, ax=ax, label="Ratio (pred/real gradient energy)")
    for layer in range(9):
        for hi in range(len(horizons)):
            val = ratio_grid[layer, hi]
            if not np.isnan(val):
                ax.text(hi, layer, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black")
    plt.tight_layout()
    path = os.path.join(save_dir, "texture_directionality_ratio.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")

    flagged = [(l, h) for l in range(9) for hi, h in enumerate(horizons)
               if not np.isnan(ratio_grid[l, hi]) and ratio_grid[l, hi] < 0.4]
    if flagged:
        print(f"[!] Possible blob collapse (ratio < 0.4) at (layer, horizon): {flagged}")
    else:
        print("No blob collapse detected (all evaluated ratios >= 0.4) at the evaluated horizons.")
    return ratio_grid


# ==============================================================================
# 5. AUTONOMOUS CONTINUOUS SIMULATION
# ==============================================================================
@torch.no_grad()
def plot_continuous_simulation(model, dataset, device, v_min, v_max, max_steps=30,
                                n_flow_steps=None, stochastic=False,
                                use_physics_guidance=False, guidance_scale=2.0,
                                divergence_threshold=150.0, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    n_flow_steps = n_flow_steps if n_flow_steps is not None else N_FLOW_STEPS

    total_available = len(dataset) if max_steps is None else min(len(dataset), max_steps)
    print(f"Running continuous simulation for up to {total_available} step(s) "
          f"(of {len(dataset)} available in the dataset)...")
    if use_physics_guidance:
        print(f"Physics-guided sampling ON (guidance_scale={guidance_scale}).")

    x_initial, _ = dataset[0]
    x_seq = x_initial.unsqueeze(0).to(device)
    fid = model._default_fidelity(1, device)
    z_buffer = model.encode_window(x_seq, fid)
    x_prev = x_seq[:, -1]
    has_trend = hasattr(dataset, "get_trend_window")

    predictions, ground_truth = [], []
    for t in range(total_available):
        _, y_t = dataset[t]
        real_t_norm = y_t[0]
        real_t = denormalize_tensor(real_t_norm, v_min, v_max)

        trend_vec = dataset.get_trend_window(t)[0].unsqueeze(0).to(device) if has_trend else None

        guidance_fn = None
        if use_physics_guidance:
            guidance_fn = make_physics_guidance(model, x_prev, fid, model.physics, dt=DT_PINN)
        z_next, _ = model.predict_next_latent(
            z_buffer, n_flow_steps=n_flow_steps, stochastic=stochastic,
            guidance_fn=guidance_fn, guidance_scale=guidance_scale, trend_vec=trend_vec,
        )
        x_next = model.decoder(z_next, fid, x_prev=x_prev, core_mask=model.core_mask)
        x_next = model.apply_texture_std_renorm(x_next)
        x_next = model.apply_texture_envelope(x_next)
        x_next = model.apply_texture_envelope_px(x_next)   # v7.20
        if trend_vec is not None:
            x_next = model.apply_trend_renorm(x_next, trend_vec)
        pred_physical = denormalize_tensor(x_next.squeeze(0), v_min, v_max).cpu()

        predictions.append(pred_physical.unsqueeze(0))
        ground_truth.append(real_t.unsqueeze(0))
        z_buffer = torch.cat([z_buffer[:, 1:], z_next.unsqueeze(1)], dim=1)
        x_prev = x_next

        if (t + 1) % 50 == 0:
            print(f"  -> processed {t + 1}/{total_available} steps...")

        mask = real_t != 0.0
        if mask.sum() > 0:
            step_mape = (torch.mean(torch.abs((real_t[mask] - pred_physical[mask]) / real_t[mask])) * 100).item()
            if step_mape > divergence_threshold:
                print(f"[!] Simulation stopped at step {t} | extreme divergence: {step_mape:.2f}%")
                break

    pred_continuous = torch.cat(predictions, dim=0).numpy()
    real_continuous = torch.cat(ground_truth, dim=0).numpy()
    actual_steps = pred_continuous.shape[0]
    time_axis = np.arange(actual_steps)

    real_core = np.where(real_continuous == 0.0, np.nan, real_continuous)
    pred_core = np.where(np.isnan(real_core), np.nan, pred_continuous)
    layer_mean_real = np.nanmean(real_core, axis=(2, 3))
    layer_mean_pred = np.nanmean(pred_core, axis=(2, 3))

    figsize = (24, 12) if max_steps is None else (18, 12)
    fig, axes = plt.subplots(3, 3, figsize=figsize, sharex=True)
    axes = axes.flatten()
    for i in range(9):
        axes[i].plot(time_axis, layer_mean_real[:, i], label="Real", color="green", linewidth=2)
        axes[i].plot(time_axis, layer_mean_pred[:, i], label="Predicted", color="red", linestyle="--", linewidth=1.5)
        axes[i].axvline(SEQ_LEN, color="black", linestyle=":", linewidth=1.2, alpha=0.8)
        if i == 0:
            axes[i].annotate(
                f"<- {SEQ_LEN} real steps | 100% feedback ->",
                xy=(SEQ_LEN, axes[i].get_ylim()[0]), xytext=(SEQ_LEN + 2, axes[i].get_ylim()[0]),
                fontsize=7, color="black", va="bottom",
            )
        axes[i].set_title(f"Layer {i}" + (" -- full evolution" if max_steps is None else ""))
        axes[i].grid(True, linestyle="--", alpha=0.7)
        axes[i].legend()
        axes[i].set_xlabel("Time step")
        axes[i].set_ylabel("Mean physical value")
    plt.tight_layout()
    suffix = "full" if max_steps is None else f"{total_available}steps"
    path = os.path.join(save_dir, f"continuous_simulation_9layers_{suffix}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


# ==============================================================================
# 6/7. LATENT SPACE DIAGNOSTICS
# ==============================================================================
@torch.no_grad()
def collect_latent_vectors(model, dataset, device, max_samples=600):
    model.eval()
    fid = model._default_fidelity(1, device)
    vectors = []
    n = min(max_samples, len(dataset))
    for i in range(n):
        x_seq, _ = dataset[i]
        z_buffer = model.encode_window(x_seq.unsqueeze(0).to(device), fid)
        vectors.append(z_buffer[:, -1].cpu().numpy())
    return np.concatenate(vectors, axis=0) if vectors else np.empty((0, model.encoder.to_latent[0].out_features))


def plot_latent_correlation(model, dataset, device, save_dir=SAVE_DIR, max_samples=600):
    os.makedirs(save_dir, exist_ok=True)
    latent_matrix = collect_latent_vectors(model, dataset, device, max_samples=max_samples)
    if latent_matrix.shape[0] == 0:
        print("[!] Empty dataset -- skipping plot_latent_correlation.")
        return
    df_latent = pd.DataFrame(latent_matrix)
    corr_matrix = df_latent.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, center=0, xticklabels=5, yticklabels=5)
    plt.title("Latent Space Correlation Matrix (high fidelity)")
    path = os.path.join(save_dir, "latent_variable_correlation.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


def plot_latent_behavior(model, dataset, device, save_dir=SAVE_DIR, max_samples=600):
    os.makedirs(save_dir, exist_ok=True)
    latent_matrix = collect_latent_vectors(model, dataset, device, max_samples=max_samples)
    if latent_matrix.shape[0] < 2:
        print("[!] Not enough samples -- skipping plot_latent_behavior.")
        return

    variances = np.var(latent_matrix, axis=0)
    top_4_indices = np.argsort(variances)[-4:][::-1]
    df_top_latent = pd.DataFrame(latent_matrix[:, top_4_indices], columns=[f"Z_{idx}" for idx in top_4_indices])
    g = sns.pairplot(df_top_latent, diag_kind="kde")
    path1 = os.path.join(save_dir, "latent_variable_vs_variable_scatter.png")
    g.savefig(path1, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path1}")

    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(latent_matrix)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent_pca[:, 0], latent_pca[:, 1], c=np.arange(len(latent_pca)), cmap="plasma", s=15)
    plt.plot(latent_pca[:, 0], latent_pca[:, 1], color="black", linewidth=0.5, alpha=0.3)
    plt.colorbar(scatter, label="Time step")
    plt.title("Phase Space: Latent Trajectory (high fidelity)")
    path2 = os.path.join(save_dir, "latent_phase_space_trajectory.png")
    plt.savefig(path2, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path2}")


# ==============================================================================
# 8. PREDICTIVE UNCERTAINTY BAND (per layer)
# ==============================================================================
@torch.no_grad()
def plot_predictive_uncertainty(model, dataset, device, v_min, v_max, n_steps=6, n_samples=8,
                                 sample_idx=0, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    x_seq_sample, y_targets_sample = dataset[sample_idx]
    x_seq = x_seq_sample.unsqueeze(0).to(device)
    trend_seq = dataset.get_trend_window(sample_idx)[:n_steps].unsqueeze(0).to(device) if hasattr(dataset, "get_trend_window") else None

    samples = []
    for _ in range(n_samples):
        preds = model.rollout(x_seq, n_steps, stochastic=True, trend_seq=trend_seq)
        samples.append(preds.cpu())
    samples = torch.cat(samples, dim=0)
    samples_real = denormalize_tensor(samples, v_min, v_max)

    mask = (y_targets_sample[:n_steps] != 0.0)
    layer_mean_samples = []
    for s in range(n_samples):
        masked = torch.where(mask, samples_real[s], torch.zeros_like(samples_real[s]))
        counts = mask.sum(dim=(-2, -1)).clamp(min=1)
        layer_mean_samples.append((masked.sum(dim=(-2, -1)) / counts).unsqueeze(0))
    layer_mean_samples = torch.cat(layer_mean_samples, dim=0).numpy()

    mean_pred = layer_mean_samples.mean(axis=0)
    std_pred = layer_mean_samples.std(axis=0)

    real_masked = torch.where(mask, y_targets_sample[:n_steps], torch.zeros_like(y_targets_sample[:n_steps]))
    real_masked = denormalize_tensor(real_masked, v_min, v_max)
    counts_real = mask.sum(dim=(-2, -1)).clamp(min=1)
    layer_mean_real = (real_masked.sum(dim=(-2, -1)) / counts_real).numpy()

    time_axis = np.arange(1, n_steps + 1)
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=True)
    axes = axes.flatten()
    for i in range(9):
        axes[i].plot(time_axis, layer_mean_real[:, i], color="green", linewidth=2, label="Real")
        axes[i].plot(time_axis, mean_pred[:, i], color="red", linestyle="--", label="Mean")
        axes[i].fill_between(time_axis, mean_pred[:, i] - std_pred[:, i], mean_pred[:, i] + std_pred[:, i],
                              color="red", alpha=0.2, label="+/-1 std")
        axes[i].set_title(f"Layer {i}")
        axes[i].grid(True, linestyle="--", alpha=0.5)
        axes[i].legend(fontsize=8)
    plt.suptitle(f"Predictive Uncertainty Band ({n_samples} stochastic rollouts)", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "predictive_uncertainty.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


# ==============================================================================
# 9. LEARNED PHYSICS COEFFICIENTS (Phase 3)
# ==============================================================================
def plot_physics_coefficients(model, ran_phase3=True, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    physics = model.physics
    v_z = physics.v_z.item()
    alpha = physics.alpha.item()
    source = physics.source.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["crimson" if s < 0 else "steelblue" for s in source]
    ax.bar(range(len(source)), source, color=colors)
    subtitle = "" if ran_phase3 else "  [!] Phase 3 did not run -- values at initialization, untrained"
    ax.set_title(
        f"Learned heat source per layer (S_i){subtitle}\n"
        f"v_z (axial velocity) = {v_z:.4f}   |   alpha (in-plane diffusivity) = {alpha:.4f}"
    )
    ax.set_xlabel("Axial layer (0=Base, 8=Top)"); ax.set_ylabel("S_i")
    ax.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    path = os.path.join(save_dir, "physics_coefficients.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


# ==============================================================================
# 10. ARCHITECTURE-SPECIFIC ABLATIONS
# ==============================================================================
@torch.no_grad()
def compute_rollout_mape_curve(model, dataset, horizon, device, v_min, v_max,
                                n_samples=30, rollout_fn=None, **rollout_kwargs):
    model.eval()
    rollout_fn = rollout_fn if rollout_fn is not None else model.rollout
    step_errors = np.zeros(horizon)
    count = 0
    has_trend = hasattr(dataset, "get_trend_window") and "trend_seq" not in rollout_kwargs
    for idx in range(min(n_samples, len(dataset))):
        x_seq, y_seq = dataset[idx]
        x_seq = x_seq.unsqueeze(0).to(device)
        y_seq = y_seq[:horizon]
        call_kwargs = rollout_kwargs
        if has_trend:
            call_kwargs = dict(rollout_kwargs, trend_seq=dataset.get_trend_window(idx)[:horizon].unsqueeze(0).to(device))
        preds = rollout_fn(x_seq, horizon, **call_kwargs)
        preds_real = denormalize_tensor(preds.squeeze(0).cpu(), v_min, v_max)
        target_real = denormalize_tensor(y_seq, v_min, v_max)
        for h in range(horizon):
            mask = target_real[h] != 0.0
            if mask.sum() == 0:
                continue
            mape = torch.mean(torch.abs((target_real[h][mask] - preds_real[h][mask]) / target_real[h][mask])) * 100
            step_errors[h] += mape.item()
        count += 1
    return step_errors / max(1, count)


def plot_mean_vs_full_ablation(model, dataset, device, v_min, v_max, horizon=15, n_samples=30, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    curve_full = compute_rollout_mape_curve(model, dataset, horizon, device, v_min, v_max,
                                             n_samples=n_samples, rollout_fn=model.rollout)
    curve_mean = compute_rollout_mape_curve(model, dataset, horizon, device, v_min, v_max,
                                             n_samples=n_samples, rollout_fn=model.rollout_mean_only)
    steps = np.arange(1, horizon + 1)
    plt.figure(figsize=(9, 5.5))
    plt.plot(steps, curve_full, marker="o", color="crimson", label="Full model (Transformer + flow residual)")
    plt.plot(steps, curve_mean, marker="s", color="steelblue", label="Deterministic mean only (no flow residual)")
    plt.xlabel("Future step (t+n)"); plt.ylabel("MAPE (%)")
    plt.title("Deterministic Backbone vs. Full Generative Model")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(save_dir, "mean_vs_full_ablation.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


def plot_flow_steps_ablation(model, dataset, device, v_min, v_max, horizon=15,
                              steps_list=(2, 4, 8, 16), n_samples=30, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(9, 5.5))
    for k in steps_list:
        curve = compute_rollout_mape_curve(model, dataset, horizon, device, v_min, v_max,
                                            n_samples=n_samples, n_flow_steps=k)
        plt.plot(np.arange(1, horizon + 1), curve, marker="o", label=f"{k} flow steps")
    plt.xlabel("Future step (t+n)"); plt.ylabel("MAPE (%)")
    plt.title("Effect of the Number of Flow-Matching Integration Steps\non Rollout Accuracy")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(save_dir, "flow_steps_ablation.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


def plot_physics_guidance_ablation(model, dataset, device, v_min, v_max, horizon=15,
                                    guidance_scales=(0.0, 1.0, 2.0, 5.0), n_samples=30, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(9, 5.5))
    for g in guidance_scales:
        curve = compute_rollout_mape_curve(
            model, dataset, horizon, device, v_min, v_max, n_samples=n_samples,
            use_physics_guidance=(g > 0), guidance_scale=g,
        )
        label = "no guidance" if g == 0 else f"guidance_scale={g}"
        plt.plot(np.arange(1, horizon + 1), curve, marker="o", label=label)
    plt.xlabel("Future step (t+n)"); plt.ylabel("MAPE (%)")
    plt.title("Effect of Inference-Time Physics Guidance on Rollout Accuracy")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(save_dir, "physics_guidance_ablation.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


@torch.no_grad()
def plot_variance_collapse_check(model, dataset, device, v_min, v_max, horizon=30, n_samples=20, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    real_stds, pred_stds = [], []
    for idx in range(min(n_samples, len(dataset))):
        x_seq, y_seq = dataset[idx]
        x_seq_b = x_seq.unsqueeze(0).to(device)
        y_seq = y_seq[:horizon]
        trend_seq = dataset.get_trend_window(idx)[:horizon].unsqueeze(0).to(device) if hasattr(dataset, "get_trend_window") else None
        preds = model.rollout(x_seq_b, horizon, trend_seq=trend_seq).squeeze(0).cpu()
        preds_real = denormalize_tensor(preds, v_min, v_max)
        target_real = denormalize_tensor(y_seq, v_min, v_max)
        mask = (target_real != 0.0).float()
        counts = mask.sum(dim=(-2, -1)).clamp(min=1)
        pred_lm = (preds_real * mask).sum(dim=(-2, -1)) / counts
        target_lm = (target_real * mask).sum(dim=(-2, -1)) / counts
        real_stds.append(target_lm.std(dim=0).numpy())
        pred_stds.append(pred_lm.std(dim=0).numpy())
    real_stds = np.stack(real_stds, axis=0).mean(axis=0)
    pred_stds = np.stack(pred_stds, axis=0).mean(axis=0)

    x = np.arange(9)
    width = 0.35
    plt.figure(figsize=(9, 5.5))
    plt.bar(x - width / 2, real_stds, width, label="Real", color="green")
    plt.bar(x + width / 2, pred_stds, width, label="Predicted", color="red", alpha=0.8)
    plt.xlabel("Layer"); plt.ylabel("Temporal std over rollout (physical units)")
    plt.title(f"Variance-Collapse Check: Predicted vs. Real Temporal Std\n"
              f"(averaged over {n_samples} rollouts, horizon={horizon})")
    plt.xticks(x)
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    path = os.path.join(save_dir, "variance_collapse_check.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")

    ratio = pred_stds / (real_stds + 1e-8)
    print("Per-layer variance ratio (predicted/real):", np.round(ratio, 3))
    if np.mean(ratio) < 0.3:
        print("[!] Predicted variance is much lower than real across layers -- "
              "consistent with the 'predict near-constant' collapse.")
    return real_stds, pred_stds


# ==============================================================================
# 10b. IS THE SPATIAL BIFURCATION A FIXED BIAS, OR CONTENT-DEPENDENT?
# ==============================================================================
@torch.no_grad()
def check_bifurcation_is_static(model, dataset, device, horizon=15, n_samples=5, layer=4, save_dir=SAVE_DIR):
    """v7.17 -- defensive bugfix: this function needs at least 2 samples (it
    compares pairs of windows with different content) -- if the available
    dataset (e.g. `long_horizon_test_ds` with a very long horizon, leaving
    few possible windows) has fewer than 2, it exits with a warning instead
    of failing. Also fixes `axes` for the n_samples==1 case (matplotlib
    doesn't return a 2D axes grid with a single column)."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    n_samples = min(n_samples, len(dataset))
    if n_samples < 2:
        print(f"[!] Only {n_samples} window(s) available for horizon={horizon} -- "
              f"at least 2 are needed to compare different content. Skipping this check "
              f"(reduce the horizon or use a dataset with more available windows).")
        return None, None

    pred_fields, real_fields = [], []
    for idx in range(n_samples):
        x_seq, y_seq = dataset[idx]
        x_seq_b = x_seq.unsqueeze(0).to(device)
        trend_seq = (dataset.get_trend_window(idx)[:horizon].unsqueeze(0).to(device)
                     if hasattr(dataset, "get_trend_window") else None)
        preds = model.rollout(x_seq_b, horizon, trend_seq=trend_seq).squeeze(0).cpu()
        pred_fields.append(preds[horizon - 1, layer].numpy())
        real_fields.append(y_seq[horizon - 1, layer].numpy())

    def flat_corr(a, b, mask):
        av, bv = a[mask], b[mask]
        av = av - av.mean(); bv = bv - bv.mean()
        denom = np.sqrt((av ** 2).sum()) * np.sqrt((bv ** 2).sum()) + 1e-8
        return float((av * bv).sum() / denom)

    mask = real_fields[0] != 0.0
    pred_cross_corrs, real_cross_corrs = [], []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            pred_cross_corrs.append(flat_corr(pred_fields[i], pred_fields[j], mask))
            real_cross_corrs.append(flat_corr(real_fields[i], real_fields[j], mask))

    fig, axes = plt.subplots(2, n_samples, figsize=(3 * n_samples, 6))
    for i in range(n_samples):
        axes[0, i].imshow(np.where(mask, real_fields[i], np.nan), cmap="viridis")
        axes[0, i].set_title(f"Real, sample {i}"); axes[0, i].axis("off")
        axes[1, i].imshow(np.where(mask, pred_fields[i], np.nan), cmap="viridis")
        axes[1, i].set_title(f"Pred, sample {i}"); axes[1, i].axis("off")
    plt.suptitle(f"Layer {layer} at t+{horizon}: does the predicted pattern change across different inputs?")
    plt.tight_layout()
    path = os.path.join(save_dir, "check_bifurcation_static.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")

    mean_pred_corr = float(np.mean(pred_cross_corrs)) if pred_cross_corrs else float("nan")
    mean_real_corr = float(np.mean(real_cross_corrs)) if real_cross_corrs else float("nan")
    print(f"AVERAGE cross-correlation across {n_samples} samples of DIFFERENT input -- "
          f"Predicted: {mean_pred_corr:+.3f} | Real: {mean_real_corr:+.3f}")
    if mean_pred_corr > 0.9 and (mean_pred_corr - mean_real_corr) > 0.3:
        print("[!] The predicted pattern is ALMOST IDENTICAL across different inputs -- "
              "evidence of a FIXED bias in the decoder, NOT of content-dependent instability.")
    else:
        print("The predicted pattern varies across inputs comparably to how real patterns vary -- "
              "no evidence of a fixed bias.")
    return mean_pred_corr, mean_real_corr


# ==============================================================================
# 10c. CENTROID AND SPREAD TRACKING
# ==============================================================================
@torch.no_grad()
def compute_centroid_and_spread(field, mask):
    rows, cols = np.where(mask)
    vals = field[mask]
    w = vals - vals.min()
    w_sum = w.sum()
    if w_sum < 1e-8:
        return float(rows.mean()), float(cols.mean()), 0.0
    r_c = float((w * rows).sum() / w_sum)
    c_c = float((w * cols).sum() / w_sum)
    spread = float(np.sqrt((w * ((rows - r_c) ** 2 + (cols - c_c) ** 2)).sum() / w_sum))
    return r_c, c_c, spread


@torch.no_grad()
def plot_centroid_tracking(model, dataset, device, v_min, v_max, horizon=200, sample_idx=0,
                            layers=(0, 1, 2, 3), save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    x_seq, y_seq = dataset[sample_idx]
    x_seq_b = x_seq.unsqueeze(0).to(device)
    horizon = min(horizon, y_seq.shape[0])
    trend_seq = (dataset.get_trend_window(sample_idx)[:horizon].unsqueeze(0).to(device)
                 if hasattr(dataset, "get_trend_window") else None)
    preds = model.rollout(x_seq_b, horizon, trend_seq=trend_seq).squeeze(0).cpu()
    preds_real = denormalize_tensor(preds, v_min, v_max).numpy()
    target_real = denormalize_tensor(y_seq[:horizon], v_min, v_max).numpy()
    mask_full = (target_real[0] != 0.0)

    fig, axes = plt.subplots(len(layers), 3, figsize=(15, 3.6 * len(layers)))
    if len(layers) == 1:
        axes = axes.reshape(1, 3)
    time_axis = np.arange(horizon)

    for li, layer in enumerate(layers):
        mask = mask_full[layer]
        real_r, real_c, real_s = [], [], []
        pred_r, pred_c, pred_s = [], [], []
        for t in range(horizon):
            r, c, s = compute_centroid_and_spread(target_real[t, layer], mask)
            real_r.append(r); real_c.append(c); real_s.append(s)
            r, c, s = compute_centroid_and_spread(preds_real[t, layer], mask)
            pred_r.append(r); pred_c.append(c); pred_s.append(s)

        axes[li, 0].plot(real_c, real_r, color="green", linewidth=1.2, alpha=0.7, label="Real")
        axes[li, 0].plot(pred_c, pred_r, color="red", linewidth=1.2, alpha=0.7, linestyle="--", label="Predicted")
        axes[li, 0].scatter([real_c[0]], [real_r[0]], color="green", marker="o", s=60, zorder=5)
        axes[li, 0].scatter([pred_c[0]], [pred_r[0]], color="red", marker="x", s=60, zorder=5)
        axes[li, 0].set_title(f"Layer {layer}: hot-spot centroid trajectory\n(col, row) over {horizon} steps")
        axes[li, 0].set_xlabel("column"); axes[li, 0].set_ylabel("row")
        axes[li, 0].legend(fontsize=7); axes[li, 0].invert_yaxis()

        dist = np.sqrt((np.array(real_r) - np.array(pred_r)) ** 2 + (np.array(real_c) - np.array(pred_c)) ** 2)
        axes[li, 1].plot(time_axis, dist, color="darkorange")
        axes[li, 1].set_title(f"Layer {layer}: centroid drift distance (pixels)\nvs. time")
        axes[li, 1].set_xlabel("Time step"); axes[li, 1].set_ylabel("Distance (pixels)")
        axes[li, 1].grid(True, linestyle="--", alpha=0.5)

        axes[li, 2].plot(time_axis, real_s, color="green", label="Real")
        axes[li, 2].plot(time_axis, pred_s, color="red", linestyle="--", label="Predicted")
        axes[li, 2].set_title(f"Layer {layer}: hot-spot spread (effective radius)\nvs. time")
        axes[li, 2].set_xlabel("Time step"); axes[li, 2].set_ylabel("Radius (pixels)")
        axes[li, 2].legend(fontsize=7); axes[li, 2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(save_dir, "centroid_spread_tracking.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


# ==============================================================================
# 10d. WHAT TRIGGERS CENTROID JUMPS?
# ==============================================================================
@torch.no_grad()
def plot_transition_trigger_analysis(model, dataset, device, v_min, v_max, horizon=1000,
                                      sample_idx=0, layer=1, max_lag=15, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    x_seq, y_seq = dataset[sample_idx]
    x_seq_b = x_seq.unsqueeze(0).to(device)
    horizon = min(horizon, y_seq.shape[0])
    has_trend = hasattr(dataset, "get_trend_window")
    if not has_trend:
        print("[!] The dataset doesn't expose a trend -- this analysis can't be run.")
        return None
    trend_seq = dataset.get_trend_window(sample_idx)[:horizon].unsqueeze(0).to(device)
    preds = model.rollout(x_seq_b, horizon, trend_seq=trend_seq).squeeze(0).cpu()
    preds_real = denormalize_tensor(preds, v_min, v_max).numpy()
    target_real = denormalize_tensor(y_seq[:horizon], v_min, v_max).numpy()
    mask = (target_real[0, layer] != 0.0)

    real_r, real_c = [], []
    pred_r, pred_c = [], []
    for t in range(horizon):
        r, c, _ = compute_centroid_and_spread(target_real[t, layer], mask)
        real_r.append(r); real_c.append(c)
        r, c, _ = compute_centroid_and_spread(preds_real[t, layer], mask)
        pred_r.append(r); pred_c.append(c)
    drift = np.sqrt((np.array(real_r) - np.array(pred_r)) ** 2 + (np.array(real_c) - np.array(pred_c)) ** 2)

    trend_np = trend_seq.squeeze(0).cpu().numpy()
    trend_rate = np.zeros(horizon)
    trend_rate[1:] = np.linalg.norm(trend_np[1:] - trend_np[:-1], axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(np.arange(horizon), drift, color="darkorange")
    axes[0].set_title(f"Layer {layer}: centroid drift distance vs. time")
    axes[0].set_ylabel("Drift (pixels)"); axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[1].plot(np.arange(horizon), trend_rate, color="steelblue")
    axes[1].set_title("Trend rate of change |d(trend)/dt| vs. time (same rollout)")
    axes[1].set_xlabel("Time step"); axes[1].set_ylabel("|d(trend)/dt|")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(save_dir, "transition_trigger_analysis.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")

    def normalized(a):
        return (a - a.mean()) / (a.std() + 1e-8)
    d_n, r_n = normalized(drift), normalized(trend_rate)
    lags = range(-max_lag, max_lag + 1)
    corrs = []
    for lag in lags:
        if lag >= 0:
            a, b = d_n[lag:], r_n[:len(r_n) - lag]
        else:
            a, b = d_n[:len(d_n) + lag], r_n[-lag:]
        corrs.append(float((a * b).mean()) if len(a) > 1 else 0.0)
    best_idx = int(np.argmax(corrs))
    best_lag, best_corr = list(lags)[best_idx], corrs[best_idx]
    print(f"Maximum correlation between |d(trend)/dt| and centroid drift: {best_corr:+.3f} at lag {best_lag}.")
    return drift, trend_rate, best_lag, best_corr


# ==============================================================================
# 10e. TOP-5 / BOTTOM-5 MAPE PIXELS PER LAYER
# ==============================================================================
@torch.no_grad()
def report_extreme_mape_pixels(model, dataset, sample_idx, device, v_min, v_max, n_steps=15,
                                top_k=5, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    x_seq, y_seq = dataset[sample_idx]
    x_seq_b = x_seq.unsqueeze(0).to(device)
    trend_seq = (dataset.get_trend_window(sample_idx)[:n_steps].unsqueeze(0).to(device)
                 if hasattr(dataset, "get_trend_window") else None)
    preds = model.rollout(x_seq_b, n_steps, trend_seq=trend_seq).squeeze(0).cpu()
    pred_real = denormalize_tensor(preds[n_steps - 1], v_min, v_max).numpy()
    real_real = denormalize_tensor(y_seq[n_steps - 1], v_min, v_max).numpy()

    report = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for layer in range(9):
            mask = real_real[layer] != 0.0
            if mask.sum() == 0:
                continue
            mape_map = np.full_like(real_real[layer], np.nan)
            mape_map[mask] = np.abs((real_real[layer][mask] - pred_real[layer][mask]) / real_real[layer][mask]) * 100
            flat_idx_sorted = np.argsort(np.where(mask, mape_map, -np.inf).ravel())[::-1]
            worst_idx = flat_idx_sorted[:top_k]
            flat_idx_sorted_asc = np.argsort(np.where(mask, mape_map, np.inf).ravel())
            best_idx = flat_idx_sorted_asc[:top_k]

            def _rows(idx_list):
                rows = []
                for flat in idx_list:
                    r, c = np.unravel_index(flat, mape_map.shape)
                    rows.append((int(r), int(c), float(real_real[layer, r, c]),
                                 float(pred_real[layer, r, c]), float(mape_map[r, c])))
                return rows

            report[layer] = {"worst": _rows(worst_idx), "best": _rows(best_idx)}

    print(f"\n=== Extreme MAPE pixels per layer (t+{n_steps}, sample {sample_idx}) ===")
    for layer in range(9):
        if layer not in report:
            continue
        print(f"\nLayer {layer}:")
        print(f"  {'':6s} {'(row,col)':>10s} | {'real':>10s} | {'pred':>10s} | {'MAPE %':>8s}")
        print(f"  Worst {top_k}:")
        for r, c, real_v, pred_v, mape_v in report[layer]["worst"]:
            print(f"    {'':4s}({r:2d},{c:2d})   | {real_v:10.4f} | {pred_v:10.4f} | {mape_v:8.3f}")
        print(f"  Best {top_k}:")
        for r, c, real_v, pred_v, mape_v in report[layer]["best"]:
            print(f"    {'':4s}({r:2d},{c:2d})   | {real_v:10.4f} | {pred_v:10.4f} | {mape_v:8.3f}")

    return report


# ==============================================================================
# 10f. ROLLOUT VIDEO
# ==============================================================================
@torch.no_grad()
def create_rollout_video(model, dataset, device, v_min, v_max, sample_idx=0, n_steps=100,
                          layers=tuple(range(9)), frame_stride=1, fps=10, save_dir=SAVE_DIR,
                          filename="rollout_video_first100.gif"):
    """v7.33 -- UPDATED defaults to match the explicit request in the model
    script's v7.33 addendum ("video of the first 100 steps... all 9
    layers, one frame per step, none skipped") -- see CHANGELOG.md v7.33.
    Previously (v7.25 baseline): `n_steps=200, layers=(0, 4, 8),
    frame_stride=2`. Real | Predicted | MAPE(%) panels, animated frame by
    frame, to inspect in detail the ring-formation transition reported
    starting around step 10."""
    import matplotlib.animation as animation

    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    x_seq, y_seq = dataset[sample_idx]
    x_seq_b = x_seq.unsqueeze(0).to(device)
    n_steps = min(n_steps, y_seq.shape[0])
    trend_seq = (dataset.get_trend_window(sample_idx)[:n_steps].unsqueeze(0).to(device)
                 if hasattr(dataset, "get_trend_window") else None)
    preds = model.rollout(x_seq_b, n_steps, trend_seq=trend_seq).squeeze(0).cpu()
    preds_real = denormalize_tensor(preds, v_min, v_max).numpy()
    real_real = denormalize_tensor(y_seq[:n_steps], v_min, v_max).numpy()

    frame_indices = list(range(0, n_steps, frame_stride))
    n_rows = len(layers)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9.5, 3.1 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    panels = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for row, layer in enumerate(layers):
            real_l = np.where(real_real[:, layer] == 0.0, np.nan, real_real[:, layer])
            pred_l = np.where(np.isnan(real_l), np.nan, preds_real[:, layer])
            vmin, vmax = np.nanmin(real_l), np.nanmax(real_l)
            im0 = axes[row, 0].imshow(real_l[0], vmin=vmin, vmax=vmax, cmap="viridis")
            im1 = axes[row, 1].imshow(pred_l[0], vmin=vmin, vmax=vmax, cmap="viridis")
            mape0 = np.abs((real_l[0] - pred_l[0]) / real_l[0]) * 100
            im2 = axes[row, 2].imshow(mape0, cmap="magma", vmin=0)
            for ax in axes[row]:
                ax.set_xticks([]); ax.set_yticks([])
            axes[row, 0].set_ylabel(f"Layer {layer}", fontsize=9)
            if row == 0:
                axes[row, 0].set_title("Real")
                axes[row, 1].set_title("Predicted")
                axes[row, 2].set_title("MAPE (%)")
            panels.append((im0, im1, im2, real_l, pred_l))

    suptitle = fig.suptitle(f"t+{1}", y=1.02)

    def update(i):
        t = frame_indices[i]
        artists = [suptitle]
        for im0, im1, im2, real_l, pred_l in panels:
            im0.set_data(real_l[t])
            im1.set_data(pred_l[t])
            with np.errstate(divide="ignore", invalid="ignore"):
                mape_t = np.abs((real_l[t] - pred_l[t]) / real_l[t]) * 100
            im2.set_data(mape_t)
            artists += [im0, im1, im2]
        suptitle.set_text(f"t+{t + 1}")
        return artists

    ani = animation.FuncAnimation(fig, update, frames=len(frame_indices), blit=False)
    path = os.path.join(save_dir, filename)
    ani.save(path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Saved: {path} ({len(frame_indices)} frames, layers {layers}, every {frame_stride} step(s) "
          f"of a {n_steps}-step rollout)")
    return path


# ==============================================================================
# 10g. STATISTICAL COMPARISON (SPECTRUM + HISTOGRAM) -- v7.17, NEW
# ==============================================================================
@torch.no_grad()
def plot_turbulence_statistics_check(model, dataset, device, v_min, v_max, layer=0,
                                      horizon=1000, n_samples=15, save_dir=SAVE_DIR):
    """v7.17 -- NEW. For turbulent layers (Layer 0 is the main case in this
    project) two physically valid fields DECORRELATE pixel by pixel at long
    horizons even though both are realistic -- the v7.9/v7.14 addenda
    already anticipated this ("the fairest evaluation at those horizons
    would be statistical... the same way any LES/DNS is evaluated against
    observations") but it had never been implemented. This function
    compares, averaged over `n_samples` independent rollouts at the
    requested horizon:
      (a) the 2D spatial power spectrum (squared FFT magnitude of the
          deviation from the layer average) -- does the model have the
          correct amount of energy at each spatial scale?, and
      (b) the histogram of pixel values inside the mask -- does the model
          have the correct statistical value distribution?
    A model that captures the PHYSICS well (even without matching pixel by
    pixel at that horizon) should match both distributions closely, even if
    the specific field doesn't coincide point by point with the real
    one."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    real_specs, pred_specs = [], []
    real_vals_all, pred_vals_all = [], []
    n_used = 0
    for idx in range(min(n_samples, len(dataset))):
        x_seq, y_seq = dataset[idx]
        this_h = min(horizon, y_seq.shape[0])
        if this_h < 1:
            continue
        x_seq_b = x_seq.unsqueeze(0).to(device)
        trend_seq = (dataset.get_trend_window(idx)[:this_h].unsqueeze(0).to(device)
                     if hasattr(dataset, "get_trend_window") else None)
        preds = model.rollout(x_seq_b, this_h, trend_seq=trend_seq).squeeze(0).cpu()
        pred_real = denormalize_tensor(preds[this_h - 1], v_min, v_max).numpy()
        real_real = denormalize_tensor(y_seq[this_h - 1], v_min, v_max).numpy()
        mask = real_real[layer] != 0.0
        if mask.sum() < 4:
            continue

        def spec2d(field):
            dev = np.where(mask, field - field[mask].mean(), 0.0)
            return np.abs(np.fft.fftshift(np.fft.fft2(dev))) ** 2

        real_specs.append(spec2d(real_real[layer]))
        pred_specs.append(spec2d(pred_real[layer]))
        real_vals_all.append(real_real[layer][mask])
        pred_vals_all.append(pred_real[layer][mask])
        n_used += 1

    if n_used == 0:
        print("[!] Not enough valid samples for the statistical comparison.")
        return None

    real_spec_avg = np.mean(real_specs, axis=0)
    pred_spec_avg = np.mean(pred_specs, axis=0)
    real_vals = np.concatenate(real_vals_all)
    pred_vals = np.concatenate(pred_vals_all)

    H, W = real_spec_avg.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    max_r = r.max()
    real_radial = np.array([real_spec_avg[r == k].mean() for k in range(max_r + 1)])
    pred_radial = np.array([pred_spec_avg[r == k].mean() for k in range(max_r + 1)])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    im0 = axes[0].imshow(np.log10(real_spec_avg + 1e-8), cmap="viridis")
    axes[0].set_title(f"Real: log-power spectrum (L{layer}, t+{horizon}, avg of {n_used})")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(np.log10(pred_spec_avg + 1e-8), cmap="viridis")
    axes[1].set_title(f"Predicted: log-power spectrum (L{layer}, t+{horizon})")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    axes[2].plot(real_radial, label="Real", color="green")
    axes[2].plot(pred_radial, label="Predicted", color="red", linestyle="--")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Spatial frequency (radial bin)")
    axes[2].set_ylabel("Power (log scale)")
    axes[2].set_title("Radially-averaged power spectrum")
    axes[2].legend(); axes[2].grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(save_dir, f"turbulence_spectrum_L{layer}_t{horizon}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.hist(real_vals, bins=40, alpha=0.5, label="Real", color="green", density=True)
    ax2.hist(pred_vals, bins=40, alpha=0.5, label="Predicted", color="red", density=True)
    ax2.set_xlabel("Physical value"); ax2.set_ylabel("Density")
    ax2.set_title(f"Pixel-value distribution (L{layer}, t+{horizon}, {n_used} samples)")
    ax2.legend(); ax2.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    path2 = os.path.join(save_dir, f"turbulence_histogram_L{layer}_t{horizon}.png")
    plt.savefig(path2, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path2}")

    print("Interpretation: if the predicted spectrum and histogram resemble the real ones "
          "(even if the specific field doesn't match pixel by pixel), the generated turbulent "
          "dynamics is statistically realistic -- the correct evaluation standard at these "
          "horizons, instead of demanding a low pixel-wise MAPE.")
    return real_radial, pred_radial, real_vals, pred_vals


# ==============================================================================
# 10h. HOW MANY DENSE POINTS ARE THERE -- v7.18, NEW (user's idea)
# ==============================================================================
def _count_density_peaks(field, mask, rel_threshold=0.5):
    """field, mask: (H,W). Counts LOCAL MAXIMA (8-neighbors) of the
    deviation from the layer average, above `rel_threshold` * std of that
    deviation, inside the mask. A layer with a SINGLE coherent focus (e.g.
    L1-L4 in this project) should give ~1; a turbulent/dispersed layer (no
    "general point," several foci that shift place) should give more than
    1 -- exactly the user's question: "how many dense points are there"
    per layer."""
    vals = field[mask]
    if vals.size < 4:
        return 0, []
    dev = np.where(mask, field - vals.mean(), -np.inf)
    thresh = rel_threshold * vals.std()
    H, W = field.shape
    count = 0
    peak_coords = []
    for i in range(H):
        for j in range(W):
            if not mask[i, j] or dev[i, j] < thresh:
                continue
            is_peak = True
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W and mask[ni, nj]:
                        if dev[ni, nj] > dev[i, j]:
                            is_peak = False
                            break
                if not is_peak:
                    break
            if is_peak:
                count += 1
                peak_coords.append((i, j))
    return count, peak_coords


@torch.no_grad()
def plot_density_peaks_check(model, dataset, device, v_min, v_max,
                              layers=(0, 1, 2, 3, 4), horizons=(15, 100, 500, 1000),
                              n_samples=10, rel_threshold=0.5, save_dir=SAVE_DIR):
    """v7.18 -- NEW. For each layer and horizon, counts how many "dense
    points" (significant local maxima) the REAL field has vs. the
    PREDICTED one, averaged over `n_samples` independent rollouts.
    Directly answers the user's question ("how many dense points are
    there") and exposes whether the model captures the correct number of
    foci -- a single coherent focus where reality has one, or a dispersed
    pattern where reality is genuinely turbulent/multi-focus -- instead of
    assuming there is always "a" hot spot (which the centroid, by
    construction, always computes, whether or not that's meaningful)."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    max_h = max(horizons)
    real_counts = {h: {l: [] for l in layers} for h in horizons}
    pred_counts = {h: {l: [] for l in layers} for h in horizons}

    for idx in range(min(n_samples, len(dataset))):
        x_seq, y_seq = dataset[idx]
        this_max_h = min(max_h, y_seq.shape[0])
        if this_max_h < 1:
            continue
        x_seq_b = x_seq.unsqueeze(0).to(device)
        trend_seq = (dataset.get_trend_window(idx)[:this_max_h].unsqueeze(0).to(device)
                     if hasattr(dataset, "get_trend_window") else None)
        preds = model.rollout(x_seq_b, this_max_h, trend_seq=trend_seq).squeeze(0).cpu()
        preds_real = denormalize_tensor(preds, v_min, v_max).numpy()
        real_real = denormalize_tensor(y_seq[:this_max_h], v_min, v_max).numpy()
        for h in horizons:
            if h > this_max_h:
                continue
            for layer in layers:
                mask = real_real[h - 1, layer] != 0.0
                if mask.sum() < 4:
                    continue
                rc, _ = _count_density_peaks(real_real[h - 1, layer], mask, rel_threshold)
                pc, _ = _count_density_peaks(preds_real[h - 1, layer], mask, rel_threshold)
                real_counts[h][layer].append(rc)
                pred_counts[h][layer].append(pc)

    fig, axes = plt.subplots(1, len(horizons), figsize=(4.2 * len(horizons), 4.2), sharey=True)
    if len(horizons) == 1:
        axes = [axes]
    x = np.arange(len(layers))
    width = 0.35
    for hi, h in enumerate(horizons):
        real_avg = [np.mean(real_counts[h][l]) if real_counts[h][l] else np.nan for l in layers]
        pred_avg = [np.mean(pred_counts[h][l]) if pred_counts[h][l] else np.nan for l in layers]
        axes[hi].bar(x - width / 2, real_avg, width, label="Real", color="green")
        axes[hi].bar(x + width / 2, pred_avg, width, label="Predicted", color="red", alpha=0.8)
        axes[hi].set_xticks(x); axes[hi].set_xticklabels([f"L{l}" for l in layers])
        axes[hi].set_title(f"t+{h}")
        axes[hi].grid(True, linestyle="--", alpha=0.4, axis="y")
        if hi == 0:
            axes[hi].set_ylabel("Number of dense points (average)")
            axes[hi].legend(fontsize=8)
    plt.suptitle(f"How many dense points per layer (threshold={rel_threshold} std, "
                 f"average of {n_samples} rollouts)", y=1.03)
    plt.tight_layout()
    path = os.path.join(save_dir, "density_peaks_check.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")
    print("Interpretation: a real value ~1 with a similar predicted value = the layer has ONE "
          "coherent focus and the model replicates it well. A real value >1 (turbulence/multi-focus) "
          "with a very different predicted value = the model is over- (or under-) simplifying the "
          "real number of foci.")
    return real_counts, pred_counts


# ==============================================================================
# 10i. FULL-CYCLE REPLICATION -- v7.18, NEW (user's idea)
# ==============================================================================
def estimate_dominant_period(series_1d, min_period=50, max_period=800,
                              expected_period_hint=None, hint_tolerance=0.3):
    """Autocorrelation over a 1D series (e.g. a layer's spatial average over
    time) to estimate the dominant oscillation period DIRECTLY from the
    data.

    v7.25 -- BUGFIX (found in a real run: the estimated period came out
    exactly `min_period`, the unambiguous signature of the bug). Cause:
    this series has a slow, low-frequency trend (visible in
    `plot_continuous_simulation` over thousands of steps) -- the ACF of a
    trending signal decays MONOTONICALLY within the search window, with no
    genuine peak; taking `argmax` then returns the SMALLEST allowed lag.
    Two detrending approaches were tried before fixing this: a moving
    average via convolution turned out to be SENSITIVE to window size (a
    small window doesn't remove the trend, a large window suffers severe
    edge effects from `np.convolve`'s implicit padding and ends up
    destroying the very oscillation being searched for). DIFFERENCING the
    series (`np.diff`) was chosen instead -- no window to tune, no edge
    effects -- plus searching for the MOST PROMINENT local peak (not the
    first, not the global max) within the search window.

    Also, `expected_period_hint` (optional): if the real period is
    approximately known from prior analysis (this project established
    ~530 steps via DMD in earlier versions), the local peak closest to that
    value (within a relative `hint_tolerance`) is prioritized over the
    globally most prominent peak -- a real, noisy series can have spurious
    peaks of larger amplitude than the genuine physical period; the hint
    makes detection much more robust while still being a real VERIFICATION
    against the data (it still searches the data, it doesn't impose the
    value)."""
    s = np.asarray(series_1d, dtype=np.float64)
    n = len(s)
    if n < 2 * min_period:
        return None
    s_detrend = np.diff(s)
    s_detrend = s_detrend - s_detrend.mean()
    n2 = len(s_detrend)
    acf = np.correlate(s_detrend, s_detrend, mode="full")[n2 - 1:]
    denom = acf[0] if acf[0] != 0 else 1e-12
    acf = acf / denom
    hi = min(max_period, len(acf))
    if hi <= min_period + 2:
        return None
    search = acf[min_period:hi]
    peaks = []
    for k in range(1, len(search) - 1):
        if search[k] > search[k - 1] and search[k] >= search[k + 1]:
            peaks.append((search[k], k))
    if not peaks:
        best_lag = min_period + int(np.argmax(search))
        print(f"  (no genuine local peaks -- using the window's global max: lag={best_lag})")
        return best_lag
    peaks.sort(reverse=True)
    top5 = [(min_period + l, round(float(v), 3)) for v, l in peaks[:5]]
    print(f"  Candidate local peaks (lag, correlation), top 5: {top5}")
    if expected_period_hint is not None:
        lo, hi_h = expected_period_hint * (1 - hint_tolerance), expected_period_hint * (1 + hint_tolerance)
        near_hint = [(v, l) for v, l in peaks if lo <= (min_period + l) <= hi_h]
        if near_hint:
            near_hint.sort(reverse=True)
            best_lag = min_period + near_hint[0][1]
            print(f"  Chose the candidate closest to the expected hint ({expected_period_hint}): lag={best_lag}")
            return best_lag
        print(f"  [!] No local peak within +/-{hint_tolerance:.0%} of the expected hint ({expected_period_hint}) "
              f"-- using the globally most prominent peak instead.")
    best_lag = min_period + peaks[0][1]
    return best_lag


@torch.no_grad()
def plot_full_cycle_replication_check(model, dataset, device, v_min, v_max, layer_for_period=0,
                                       n_cycles=3, expected_period_hint=530, save_dir=SAVE_DIR):
    """v7.18 -- NEW (user's idea: "cut cycles across all bases to see if the
    model can generally replicate those steps"). Estimates the dominant
    oscillation period DIRECTLY from `dataset`'s real data (autocorrelation
    on a layer's spatial average), and picks `n_cycles` starting points
    spaced approximately one period apart -- runs a free rollout of ONE
    FULL CYCLE from each and compares it against reality. A harder, more
    interpretable test than an arbitrary horizon: it verifies whether the
    model can reproduce a recognizable PHYSICAL cycle from start to finish,
    not just an arbitrary stretch -- a direct signal of whether it
    "understands" the physics or just interpolates locally.

    v7.25 -- `expected_period_hint=530`: a real run revealed a detection bug
    (the estimated period came out exactly the allowed minimum, see
    `estimate_dominant_period`). The default hint (~530 steps) comes from
    this same project's earlier DMD analysis -- it remains a VERIFICATION
    on `dataset`'s data (the peak must genuinely exist in the ACF, only the
    choice among genuine candidates is guided), not an imposed value. Pass
    `expected_period_hint=None` for the original blind detection."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    series = []
    max_scan = min(len(dataset), 4000)   # enough for several periods without scanning the whole (80%, potentially huge) test set
    for i in range(max_scan):
        _, y = dataset[i]
        series.append(masked_spatial_mean_series(y[0].unsqueeze(0))[0, layer_for_period].item())
    series = np.array(series)
    period = estimate_dominant_period(series, expected_period_hint=expected_period_hint)
    if period is None:
        print("[!] Could not estimate a dominant period (too little data) -- skipping this check.")
        return None
    print(f"Estimated dominant period (autocorrelation, Layer {layer_for_period}): {period} steps.")

    starts = [i * period for i in range(n_cycles) if i * period + period < len(dataset)]
    if not starts:
        print("[!] The dataset doesn't have enough steps for even one full cycle -- skipping.")
        return None

    fig, axes = plt.subplots(len(starts), 1, figsize=(12, 3.2 * len(starts)), sharex=True)
    if len(starts) == 1:
        axes = [axes]
    for ci, start in enumerate(starts):
        x_seq, y_seq = dataset[start]
        x_seq_b = x_seq.unsqueeze(0).to(device)
        this_h = min(period, y_seq.shape[0])
        trend_seq = (dataset.get_trend_window(start)[:this_h].unsqueeze(0).to(device)
                     if hasattr(dataset, "get_trend_window") else None)
        preds = model.rollout_mean_only(x_seq_b, this_h, trend_seq=trend_seq).squeeze(0).cpu()
        pred_real = denormalize_tensor(preds, v_min, v_max)
        real_real = denormalize_tensor(y_seq[:this_h], v_min, v_max)
        pred_series = masked_spatial_mean_series(pred_real.unsqueeze(0))[0, :, layer_for_period].numpy()
        real_series = masked_spatial_mean_series(real_real.unsqueeze(0))[0, :, layer_for_period].numpy()
        t_axis = np.arange(this_h)
        axes[ci].plot(t_axis, real_series, color="green", label="Real", linewidth=2)
        axes[ci].plot(t_axis, pred_series, color="red", linestyle="--", label="Predicted")
        axes[ci].set_title(f"Cycle {ci + 1} (start=t{start}, Layer {layer_for_period}, {this_h} steps)")
        axes[ci].legend(fontsize=8); axes[ci].grid(True, linestyle="--", alpha=0.5)
    axes[-1].set_xlabel("Steps into cycle")
    plt.tight_layout()
    path = os.path.join(save_dir, f"full_cycle_replication_L{layer_for_period}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")
    print("Interpretation: if the predicted curve (red) closely follows the real one (green) "
          "throughout the WHOLE cycle (rise, peak, fall, trough) across all 3 cuts, the model "
          "replicates recognizable physical dynamics, not just a short local interpolation.")
    return period, starts


# ==============================================================================
# 10j. CONCENTRATION FLOW-FIELD VECTOR FIELD -- v7.24, NEW (user's idea)
# ==============================================================================
def estimate_flow_field_lk(frames, mask, win=3, smooth_frames=5, rel_eps=1e-3):
    """v7.24 -- NEW. Estimates, per layer and per pixel, the vector field
    v=(vx,vy) that best explains the concentration's motion between
    consecutive frames, solving the advection equation
    (dC/dt + v.grad(C) = 0) by local least squares in a (2*win+1)^2-pixel
    window around each point -- the classic Lucas-Kanade method applied to
    the physical field. `frames`: (T,H,W) of ONE layer. Averages temporal
    gradients over `smooth_frames` consecutive pairs for noise robustness.
    Returns (vy, vx) of shape (H,W) (NaN outside the mask or where the
    local gradient is genuinely zero).

    v7.25 -- BUGFIX: the previous version rejected a pixel if
    `det(AtA) < eps` with an ABSOLUTE `eps` (1e-6) -- but AtA's magnitude
    depends on the layer's physical scale, which in this project varies by
    ORDERS OF MAGNITUDE (L0 ~ 60-110, L5-L8 ~ 77.3-77.7) -- the same
    absolute threshold systematically rejected more pixels in small-range
    layers, leaving the divergence maps nearly empty (visible in a real
    run: most of the domain came out gray/NaN). Fix: regularization is now
    PROPORTIONAL to the local scale (`rel_eps * trace(AtA)`, not an
    absolute threshold), and a pixel is only discarded if the local
    gradient is genuinely ~zero (nothing to solve), not from an
    inappropriate scale comparison."""
    T, H, W = frames.shape
    n_pairs = min(smooth_frames, T - 1)
    gx = np.zeros((H, W)); gy = np.zeros((H, W)); gt = np.zeros((H, W))
    for k in range(n_pairs):
        f0, f1 = frames[k], frames[k + 1]
        favg = 0.5 * (f0 + f1)
        gy_k, gx_k = np.gradient(favg)
        gx += gx_k; gy += gy_k; gt += (f1 - f0)
    gx /= n_pairs; gy /= n_pairs; gt /= n_pairs

    vy = np.full((H, W), np.nan); vx = np.full((H, W), np.nan)
    for i in range(H):
        for j in range(W):
            if not mask[i, j]:
                continue
            i0, i1 = max(0, i - win), min(H, i + win + 1)
            j0, j1 = max(0, j - win), min(W, j + win + 1)
            msub = mask[i0:i1, j0:j1]
            if msub.sum() < 4:
                continue
            A = np.stack([gx[i0:i1, j0:j1][msub], gy[i0:i1, j0:j1][msub]], axis=1)
            b = -gt[i0:i1, j0:j1][msub]
            AtA = A.T @ A
            trace = np.trace(AtA)
            if trace < 1e-14 * max(1.0, np.abs(frames).max()) ** 2:
                continue   # genuinely zero local gradient -- nothing to solve
            ridge = rel_eps * trace
            v = np.linalg.solve(AtA + ridge * np.eye(2), A.T @ b)
            vx[i, j], vy[i, j] = v[0], v[1]
    return vy, vx


@torch.no_grad()
def plot_concentration_flow_field(model, dataset, device, v_min, v_max, layers=(1, 2, 3, 4),
                                   start_step=0, smooth_frames=5, horizon_offset=0,
                                   sample_idx=0, save_dir=SAVE_DIR):
    """v7.24 -- NEW (implements the user's idea): for each layer, a vector
    field of where concentration FLOWS (arrows) over the real and predicted
    fields in the same time window, plus the DIVERGENCE map of that flow
    (div<0 = mass is CONCENTRATING there; div>0 = it's DISPERSING outward)
    -- exactly "where it's concentrating... how it disperses... whether
    it's increasing or decreasing." Comparing real vs. predicted arrows and
    divergence answers empirically whether the model captures physical
    TRANSPORT, not just values."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    x_seq, y_seq = dataset[sample_idx]
    need = horizon_offset + smooth_frames + 1
    this_h = min(need + 5, y_seq.shape[0])
    if this_h < need:
        print("[!] Insufficient window for the flow field -- skipping.")
        return None
    x_seq_b = x_seq.unsqueeze(0).to(device)
    trend_seq = (dataset.get_trend_window(sample_idx)[:this_h].unsqueeze(0).to(device)
                 if hasattr(dataset, "get_trend_window") else None)
    preds = model.rollout(x_seq_b, this_h, trend_seq=trend_seq).squeeze(0).cpu()
    preds_real = denormalize_tensor(preds, v_min, v_max).numpy()
    real_real = denormalize_tensor(y_seq[:this_h], v_min, v_max).numpy()

    t0 = horizon_offset
    fig, axes = plt.subplots(len(layers), 4, figsize=(19, 4.4 * len(layers)))
    if len(layers) == 1:
        axes = axes.reshape(1, 4)
    yy, xx = np.meshgrid(np.arange(real_real.shape[2]), np.arange(real_real.shape[3]), indexing="ij")

    for li, layer in enumerate(layers):
        mask = real_real[t0, layer] != 0.0
        r_vy, r_vx = estimate_flow_field_lk(real_real[t0:t0 + smooth_frames + 1, layer], mask, smooth_frames=smooth_frames)
        p_vy, p_vx = estimate_flow_field_lk(preds_real[t0:t0 + smooth_frames + 1, layer], mask, smooth_frames=smooth_frames)

        def _div(vy, vx):
            d = np.full_like(vy, np.nan)
            vyf = np.nan_to_num(vy); vxf = np.nan_to_num(vx)
            dy = np.gradient(vyf, axis=0); dx = np.gradient(vxf, axis=1)
            d[mask] = (dy + dx)[mask]
            return d

        for col, (name, field, vy, vx) in enumerate([
            ("Real field + flow", real_real[t0, layer], r_vy, r_vx),
            ("Predicted field + flow", preds_real[t0, layer], p_vy, p_vx),
        ]):
            ax = axes[li, col]
            ax.imshow(np.where(mask, field, np.nan), cmap="viridis")
            ax.quiver(xx, yy, np.nan_to_num(vx), -np.nan_to_num(vy), color="white", scale=None, width=0.004)
            ax.set_title(f"L{layer}: {name} (t+{t0 + 1}..t+{t0 + smooth_frames + 1})", fontsize=9)
            ax.axis("off")
        for col, (name, vy, vx) in enumerate([("Real divergence", r_vy, r_vx),
                                               ("Predicted divergence", p_vy, p_vx)], start=2):
            ax = axes[li, col]
            im = ax.imshow(_div(vy, vx), cmap="coolwarm", vmin=-0.5, vmax=0.5)
            ax.set_title(f"L{layer}: {name}\n(blue<0=concentrating, red>0=dispersing)", fontsize=9)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    path = os.path.join(save_dir, "concentration_flow_field.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")
    print("Interpretation: if the predicted arrows (transport direction) and divergence maps "
          "resemble the real ones, the model captures the physical MOVEMENT of the concentration, "
          "not just its values -- the direct test of the vector-field idea.")
    return path


# ==============================================================================
# 11. EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Bootstrap for standalone runs (this delivery, not in the original
    # v7.33 notebook cells). This script was originally meant to be pasted
    # into the SAME kernel session right after reactor_world_model.py
    # finished, reusing `model`, `test_ds`, `v_min`, etc. straight out of
    # that session's memory. Run as a separate terminal command instead
    # (`python3 reactor_world_model_diagnostics.py`, after a previous,
    # separate `python3 reactor_world_model.py` run), none of those
    # variables exist in this fresh process -- so if they're missing, we
    # import reactor_world_model.py as a module (this only re-runs its
    # data loading and class/function definitions, which sit at module
    # level -- NOT training, which lives inside ITS OWN
    # `if __name__ == "__main__":` guard and is never re-triggered by an
    # import) and load the checkpoint it saved at the end of training.
    if "model" not in globals():
        print("Standalone run detected (no `model` in scope) -- importing "
              "reactor_world_model.py and loading its saved checkpoint...")
        import reactor_world_model as _rwm
        globals().update({k: v for k, v in vars(_rwm).items() if not k.startswith("_")})

        _ckpt_path = os.path.join("outputs", "checkpoint_v7_33.pt")
        if not os.path.exists(_ckpt_path):
            raise FileNotFoundError(
                f"No checkpoint found at '{_ckpt_path}'. Run reactor_world_model.py first "
                f"(it saves a checkpoint there automatically once training finishes) -- or, "
                f"if you're working in a notebook, paste this script's cells into the SAME "
                f"session that just finished training instead of running it as a separate "
                f"script."
            )
        # weights_only=False: PyTorch >=2.6 defaults to True, which refuses to
        # unpickle the numpy arrays (v_min/v_max) and plain-dict training
        # histories this checkpoint also stores, not just tensors. Safe here
        # because this checkpoint is the one this SAME repo's training script
        # just wrote, not a third-party file.
        _ckpt = torch.load(_ckpt_path, map_location=device, weights_only=False)
        model = LatentWorldModelV6(LATENT_DIM).to(device)
        model.load_state_dict(_ckpt["model_state_dict"])
        model.eval()
        v_min, v_max = _ckpt["v_min"], _ckpt["v_max"]
        hist_ae_pretrain = _ckpt["hist_ae_pretrain"]
        hist_ae_finetune = _ckpt["hist_ae_finetune"]
        hist_dyn_pretrain = _ckpt["hist_dyn_pretrain"]
        hist_dyn_finetune = _ckpt["hist_dyn_finetune"]
        hist_rollout = _ckpt.get("hist_rollout")
        hist_pinn = _ckpt.get("hist_pinn")

        loaders = make_loaders()
        val_ds = loaders["val_ds"]
        test_ds = loaders["test_ds"]
        test_loader = loaders["test_loader"]
        print(f"Checkpoint loaded from '{_ckpt_path}'. Rebuilt datasets/loaders "
              f"from reactor_world_model.py's data-loading code (module-level, "
              f"deterministic given the same data/ folder -- so this reproduces the "
              f"exact same train/val/test split used during training).")
    else:
        print("Continuing in the same session that trained the model -- using "
              "`model`, `test_ds`, etc. already in scope.")


    N_STEPS_EVAL = min(getattr(test_ds, "horizon_len", 20), 15)
    print(f"Evaluating the error matrix (horizon x layers, {N_STEPS_EVAL} steps)...")
    final_errors = evaluate_error_matrix(model, test_loader, N_STEPS_EVAL, device, v_min, v_max)

    print("Generating diagnostic figures...")
    plot_training_overview(
        hist_ae_pretrain, hist_ae_finetune, hist_dyn_pretrain, hist_dyn_finetune,
        hist_rollout=globals().get("hist_rollout"), hist_pinn=globals().get("hist_pinn"),
    )
    plot_layer_mape_t1(final_errors)
    plot_error_propagation(final_errors)
    plot_spatial_error(model, test_ds, sample_idx=0, device=device, v_min=v_min, v_max=v_max, n_steps=N_STEPS_EVAL)

    LONG_HORIZONS = (10, 50, 100, 200, 500, 750, 1000)
    _long_horizon_available = max(1, len(data_test) - SEQ_LEN)
    long_horizon_test_ds = ReactorWindowDataset(
        data_test, horizon_len=min(max(LONG_HORIZONS), _long_horizon_available),
        trend_override=DMD_FORECAST_TEST,
    )
    print(f"\nEvaluating spatial error evolution at horizons {LONG_HORIZONS} "
          f"(dataset built with horizon_len={long_horizon_test_ds.horizon_len})...")
    plot_mape_evolution_multi_horizon(model, long_horizon_test_ds, sample_idx=0, device=device,
                                       v_min=v_min, v_max=v_max, horizons=LONG_HORIZONS)
    plot_spatial_error(model, long_horizon_test_ds, sample_idx=0, device=device, v_min=v_min, v_max=v_max,
                        n_steps=long_horizon_test_ds.horizon_len)

    # v7.15 -- NEW: direct blob-collapse check (invisible to MAPE in L4-L8)
    print("\nChecking texture directionality (detects blob-collapse invisible to MAPE, v7.15)...")
    plot_texture_directionality_check(model, long_horizon_test_ds, device=device, v_min=v_min, v_max=v_max,
                                       horizons=tuple(h for h in (15, 100, 500, 1000)
                                                      if h <= long_horizon_test_ds.horizon_len) or (long_horizon_test_ds.horizon_len,),
                                       n_samples=10)

    plot_continuous_simulation(model, test_ds, device=device, v_min=v_min, v_max=v_max, max_steps=30)
    plot_latent_correlation(model, test_ds, device=device)
    plot_latent_behavior(model, test_ds, device=device)
    plot_predictive_uncertainty(model, test_ds, device=device, v_min=v_min, v_max=v_max, n_steps=N_STEPS_EVAL, n_samples=8)
    plot_physics_coefficients(model, ran_phase3=RUN_PHASE3_PINN)

    print("\nRunning architecture-specific ablations (slower -- several full rollouts per curve)...")
    plot_mean_vs_full_ablation(model, test_ds, device=device, v_min=v_min, v_max=v_max, horizon=N_STEPS_EVAL, n_samples=20)
    plot_flow_steps_ablation(model, test_ds, device=device, v_min=v_min, v_max=v_max, horizon=N_STEPS_EVAL,
                              steps_list=(2, 4, 8, 16), n_samples=20)
    plot_physics_guidance_ablation(model, test_ds, device=device, v_min=v_min, v_max=v_max, horizon=N_STEPS_EVAL,
                                    guidance_scales=(0.0, 1.0, 2.0, 5.0), n_samples=20)

    print("\nChecking for mode collapse ('predict near-constant')...")
    plot_variance_collapse_check(model, test_ds, device=device, v_min=v_min, v_max=v_max,
                                  horizon=min(N_STEPS_EVAL, 30), n_samples=20)
    x_seq_flow_check, _ = test_ds[0]
    flow_report = check_flow_diversity(model, x_seq_flow_check.unsqueeze(0).to(device))
    print_flow_diversity_report(flow_report)

    print("\nChecking whether any spatial bifurcation pattern is a FIXED bias or content-dependent...")
    # v7.17 -- changed from layer=4 to layer=0 (see CHANGELOG.md v7.17): the
    # real run showed a diagonal/band pattern in Pred L0 ALMOST IDENTICAL from
    # t+10 through t+1000 -- exactly the signature this function is meant to
    # detect (content-independence = fixed architectural bias). Run at two
    # horizons: short (as before) and long (where the pattern was observed).
    check_bifurcation_is_static(model, test_ds, device=device, horizon=min(N_STEPS_EVAL, 15),
                                 n_samples=min(5, len(test_ds)), layer=0)
    check_bifurcation_is_static(model, long_horizon_test_ds, device=device,
                                 horizon=min(1000, long_horizon_test_ds.horizon_len),
                                 n_samples=min(5, len(long_horizon_test_ds)), layer=0)

    # v7.17 -- NEW: STATISTICAL comparison (power spectrum + histogram, not
    # pixel by pixel) for Layer 0 -- the "fair" evaluation for a turbulent
    # layer that the v7.9/v7.14 addenda promised but had never implemented.
    # Run at a long horizon (t+1000) where pixel-wise MAPE is less informative
    # due to genuine turbulent decorrelation.
    print("\nComparing turbulence statistics for Layer 0 (spectrum + histogram, fair evaluation "
          "for a turbulent layer where pixel-wise MAPE is not meaningful at long horizons)...")
    plot_turbulence_statistics_check(model, long_horizon_test_ds, device=device, v_min=v_min, v_max=v_max,
                                      layer=0, horizon=min(1000, long_horizon_test_ds.horizon_len), n_samples=15)

    print("\nTracking hot-spot centroid drift and spread over a long rollout (layers 0-3)...")
    plot_centroid_tracking(model, long_horizon_test_ds, device=device, v_min=v_min, v_max=v_max,
                            horizon=min(1000, long_horizon_test_ds.horizon_len), layers=(0, 1, 2, 3))

    print("\nTesting whether centroid-drift jumps are triggered by fast trend transitions...")
    plot_transition_trigger_analysis(model, long_horizon_test_ds, device=device, v_min=v_min, v_max=v_max,
                                      horizon=min(1000, long_horizon_test_ds.horizon_len), layer=1)

    print("\nReporting the 5 best / 5 worst MAPE pixels per layer...")
    report_extreme_mape_pixels(model, test_ds, sample_idx=0, device=device, v_min=v_min, v_max=v_max,
                                n_steps=N_STEPS_EVAL, top_k=5)

    # v7.33 -- UPDATED: first-100-steps video, all 9 layers, one frame per step
    # (see CHANGELOG.md v7.33 and the updated `create_rollout_video` defaults
    # above). Previously (v7.25 baseline): 200 steps, layers (0,4,8), stride 2.
    print("\nRendering rollout video (Real / Predicted / MAPE, all 9 layers, first 100 steps, "
          "no frames skipped -- v7.33)...")
    create_rollout_video(model, long_horizon_test_ds, device=device, v_min=v_min, v_max=v_max,
                          n_steps=min(100, long_horizon_test_ds.horizon_len), layers=tuple(range(9)),
                          frame_stride=1, fps=10)

    print("\nEstimating concentration flow fields (user's vector-field idea): real vs predicted transport...")
    plot_concentration_flow_field(model, long_horizon_test_ds, device=device, v_min=v_min, v_max=v_max,
                                   layers=(1, 2, 3, 4), horizon_offset=0)
    plot_concentration_flow_field(model, long_horizon_test_ds, device=device, v_min=v_min, v_max=v_max,
                                   layers=(1, 2, 3, 4),
                                   horizon_offset=max(0, min(494, long_horizon_test_ds.horizon_len - 12)))

    print("\nCounting density peaks per layer/horizon (how many distinct hot-spots, real vs. predicted)...")
    plot_density_peaks_check(model, long_horizon_test_ds, device=device, v_min=v_min, v_max=v_max,
                              layers=(0, 1, 2, 3, 4),
                              horizons=tuple(h for h in (15, 100, 500, 1000) if h <= long_horizon_test_ds.horizon_len)
                              or (long_horizon_test_ds.horizon_len,),
                              n_samples=10)

    print("\nChecking whether the model can replicate full oscillation cycles end-to-end "
          "(period estimated directly from the data, not assumed)...")
    # v7.18 -- uses `long_horizon_test_ds` (not `test_ds`): the real period
    # (~500+ steps) exceeds `test_ds`'s short horizon_len (20) --
    # `get_trend_window` would fall short for a full-cycle rollout.
    plot_full_cycle_replication_check(model, long_horizon_test_ds, device=device, v_min=v_min, v_max=v_max,
                                       layer_for_period=0, n_cycles=3)

    print(f"\nDone! Figures saved to {SAVE_DIR}/")