# Changelog — Reactor Core Latent World Model

All notable development history of the project is recorded here, in reverse-chronological
order (latest first). This file is the single source of truth for *why* each mechanism in
the code exists — the source files (`src/reactor_world_model.py`,
`src/reactor_world_model_diagnostics.py`) intentionally keep in-line comments short and
point back here, instead of duplicating this narrative.

Legend: **Evidence** = what a real training/evaluation run showed. **Fix** = what changed.
**Lesson** = the generalizable takeaway kept for future versions.

---

## v7.33 — Diffusion/neighbor-consistency loss + first-100-steps video (urgent delivery)

**Context.** The user reported the same ring artifact after the v7.32 fix and asked for the
best deliverable model the same day. Since five targeted attempts had already gone into the
ring/edge-band artifact from the statistics/envelope side (v7.26, v7.29, v7.30, v7.31,
v7.32 — one of which, v7.31, made things worse before being corrected), a sixth uncertain
adjustment to the *same* mechanism was not attempted under time pressure. Instead, this
version ships two pieces of **confirmed** value:

1. **`diffusion_consistency_loss`** (new, `W_DIFFUSION=1.0`)  — directly implements the
   user's idea: *"a fluid/thermal loss… see how each point reacts with its neighbor."* It
   encodes the residual of the **pure diffusion equation** (`dC/dt = alpha * Laplacian(C)`)
   — exactly "how each point reacts with its 4 immediate neighbors" — *without* the axial
   term between layers (that relationship is already covered by `interlayer_coupling_loss`,
   v7.24), avoiding duplicating the same mechanism twice. Unlike the optional Phase 3/PINN
   residual (which historically never fit well — v7.4: "the residual never went to zero"),
   this loss does **not** demand a zero residual: it computes the *same* residual for
   prediction and ground truth and penalizes the two differing — much safer, the same
   pattern used by `interlayer_coupling_loss`. It reuses the existing per-layer confidence
   weight (`model.centroid_layer_weight`) to automatically damp its effect on L0 — the user
   themselves noted "it wouldn't apply much on the first layer either," and that
   data-derived weight already captures exactly that distinction. Validated with unit tests
   before integration: ~0 on a field that genuinely follows the diffusion equation compared
   to itself, clearly positive against random noise that doesn't follow it, finite
   gradients, and correct per-layer scaling. `alpha` is used detached from the gradient
   graph during this phase (which does not train the physics module).
2. **First-100-steps rollout video** (diagnostics script, explicit request): Real |
   Predicted | MAPE(%), all 9 layers, one frame per step with none skipped — to inspect in
   detail the transition reported ("a ring starts forming… from step 10 onward").

**On the ring artifact:** still an **open problem**. The most likely cause, after five
statistics/envelope-side attempts without a definitive fix, is a decoder architectural bias
(coarse 4×4→8×8→15×15 bilinear upsampling, suspected since v7.17/v7.27) — an architecture
change was already tried once (CoordConv, v7.28) and caused a worse regression, reverted in
v7.29. A recommended next step (not attempted here, no time pressure): a more careful
architecture change than CoordConv — e.g., a gentler intermediate-resolution progression
(4×4→6×6→10×10→15×15 instead of one large jump) — validated with the same shape/gradient
unit tests used (unsuccessfully) for CoordConv.

---

## v7.32 — Self-correction: the v7.31 reasoning was inverted

**Evidence** (real run of v7.31): the edge artifact did *not* disappear — it *intensified*
and changed shape: a geometric ring/frame pattern became visible *inside* the field, with
bright columns on *both* edges (previously only the left). A regression, not an improvement.

**Reasoning error in v7.31:** the logic was "the whole-layer scalar is the most *robust*
estimate available, use it at the boundary instead of the noisy per-pixel percentile." This
conflates "robust" (many samples) with "appropriately scaled" (reflects the variability of
*that specific* region). The whole-layer scalar is dominated by the variability of the
domain *center* — where the hot spot moves, with much larger excursions than a static
boundary cell. Using it at the boundary doesn't tighten anything — it *widens* the
constraint well beyond what a boundary pixel actually needs, letting it oscillate *more*
freely than with its own (already generous) percentile from v7.20–v7.30. Verified with a
direct synthetic test before correcting: in a realistic scenario (high-variance center,
low-variance boundary), the "boundary = whole-layer scalar" width came out ~4.5× *wider*
than the width that genuinely corresponds to the boundary region.

**Fix:** neither a single pixel's percentile (too noisy with ~10% of the data — the original
cause) nor the whole-layer scalar (too wide — the v7.31 error): boundary-cell deviations are
*pooled* together across the entire statistics window, and the percentile is computed over
that pooled set. This gives a large sample count (all boundary cells × all frames — robust)
while staying bounded to the *actual* variability of the boundary region (appropriately
scaled). Confirmed with the same synthetic test: the pooled boundary width came out ~4.5×
*narrower* than the whole-layer scalar — the correct direction, opposite of the v7.31 error.

**Lesson:** "more samples" and "appropriate scale" are distinct properties of an estimator,
and optimizing only the first (as v7.31 did) can worsen the second. This fix optimizes both
simultaneously: a wide pool (robustness) restricted to the physically relevant region
(appropriate scale).

---

## v7.31 — Comparison against an earlier version of this same project + root-fix attempt for the boundary artifact

**Main finding:** the file the user pointed to as having "better dynamics" turned out to be
an *earlier version of this same project* (v7.14, before the strict split of v7.17) — not an
external model. Key difference: that version used 50% of the high-fidelity data to fit the
DMD trend, texture envelope, and centroid model (closed-form — SVD, percentiles — no
gradient, no overfitting risk from using more data there) and only 10% for gradient-based
fine-tuning. When v7.17 implemented the user's explicit constraint ("10% for everything"),
that statistics window shrank 5× at once — a plausible and sufficient statistical cause for
visually less-smooth dynamics, independent of any architecture change.

**Evidence linking this to the boundary artifact:** the persistent edge-of-mask band
survived *three* architecture/training-side fix attempts (v7.26 spatial smoothing, v7.29
CoordConv revert, v7.30 reflect padding + width cap) — ruling out architecture as the sole
cause. Direct inspection confirmed boundary cells are the worst-sampled in the entire
domain (fewest comparable historical configurations, resampling effects from the original
CFD mesh) — with only ~10% of the data, their per-pixel percentile (even width-capped by
v7.30) can still have a shifted *center* that no width cap corrects.

**Two changes, neither touching the network architecture:**
1. `HIGH_FIDELITY_STATS_FRAC=0.25` (new): reintroduces the distinction between a
   "statistics window" (25%, only for DMD/envelope/centroid, deterministic gradient-free
   computations) and the "gradient-training window" (10%, strict — the user's explicit
   training-budget constraint is unchanged). Test remains large (~65% instead of ~80%) — a
   small evaluation-coverage cost against the benefit of better-conditioning *all*
   data-derived statistics.
2. For boundary cells specifically (adjacent to the mask exterior, confirmed
   programmatically) the whole-layer *scalar* envelope is used instead of the per-pixel
   percentile — interior (better-sampled) cells keep their per-pixel envelope unchanged.
   *(Corrected in v7.32 — see above: this choice was evidence-directed but wrong.)*

**Scope note:** given the code's size and maturity (30+ incremental, evidence-validated
versions), a blind full rewrite risked reintroducing already-solved bugs (exactly what
happened with CoordConv in v7.28). Code changes were reserved for the two findings with
direct evidence above; broader "what's essential" analysis was handled as documentation
rather than code changes.

---

## v7.30 — CoordConv revert (v7.29) confirmed working; two softer, distinct residual issues found and fixed

**Confirmation** (real run of v7.29): no more catastrophic collapse or horizon-worsening
column — the CoordConv revert worked exactly as expected.

**Issue 1 — a boundary band persists, but with a different signature:** present already from
t+10 and *constant* across the horizon (not accumulating, unlike the v7.28 bug). This rules
out CoordConv's "non-recoverable accumulated push" mechanism and points to something
simpler: (a) "replicate" padding in the decoder's convolutions, which repeats the boundary
value outward and gives the convolution an artificially flat edge statistic (a classic,
well-documented CNN bias), possibly combined with (b) a boundary pixel whose historical
per-pixel percentile (v7.20) came out anomalously wide with only ~10% training data. Two
low-risk fixes (neither adds new capacity like CoordConv did):
1. `TEXTURE_ENV_PX_MAX_WIDTH_RATIO=1.5`: an additional cap on the (already smoothed, v7.26)
   per-pixel envelope — no pixel's envelope width may exceed 1.5× that layer's scalar
   width. Can only *narrow* anomalous pixels, never widen any.
2. `padding_mode` changed from `"replicate"` to `"reflect"` in the decoder's two
   convolutions — a hyperparameter change on an already-existing layer, no new parameters,
   the standard alternative for this specific edge bias.

**Issue 2 — the real wide band keeps shrinking to a compact blob at long horizon** (L3–L5), a
"preserve the dynamics" problem the user flagged directly. Cause: `spatial_anisotropy_loss`
(v7.25) only compared the major/minor axis *ratio* (how elongated, not how *large*) — a 2×2
blob and a 10×2 band can have similar elongation ratios even though the absolute *size* is
very wrong, so the loss was nearly satisfied even as the hot spot shrank. Fix:
1. A new absolute-scale term in `spatial_anisotropy_loss`:
   `(sqrt(lambda_major_pred) - sqrt(lambda_major_real))^2` — directly penalizes how far the
   major axis extends, complementing (not replacing) the existing ratio+orientation term.
   Validated with a unit test before integration: a shrunk blob with the *same* elongation
   ratio as reality now gives a clearly positive loss (2.55) instead of ~0 as before; a
   prediction correct in both size and shape still gives ~0; gradients are finite.
2. `CENTROID_SPREAD_SUBWEIGHT`: 0.3 → 0.5 (raised again, partially reversing the v7.22
   decrease). The risk that motivated that decrease (raising spread worsened elongation in
   v7.21) is now mitigated because a dedicated orientation/elongation loss exists that v7.21
   didn't have — raising spread no longer competes alone against shape.

None of these three changes add new capacity to the network (unlike CoordConv) — they are
an additional cap on an existing guarantee, a hyperparameter change on an existing layer, and
an extension of an already-validated loss. Much lower risk profile than v7.28.

---

## v7.29 — The v7.28 architecture experiment failed: reverted, with a full mechanistic explanation

**Evidence** (real run of v7.28): instead of fixing the block/mosaic artifact, L6 and L7
predictions *collapsed* to a nearly flat field with a single *saturated column* at the left
edge (column 0) — visible already at t+10 and increasingly severe up to t+2000 (by t+2000,
predicted L6 is entirely uniform except for that column). L3–L5 showed the same column
artifact plus collapse of the real wide band to a small, sharp blob. A *more severe*
regression than the blocking v7.28 aimed to fix.

**Mechanistic cause** (reconstructed by comparing against a reference model the user
provided, whose own internal v7.4/v7.6 documentation described an almost identical failure):
that older model had already diagnosed, at the time, "bright streaks stuck to the mask
*boundary*" and "saturated patches with hard, rectangular edges" in long rollouts — caused
by accumulated saturation of the *same* coarse bilinear upsampling (4×4→8×8→15×15) this
project still uses since v6. The v7.6 fix (tanh+clamp bounded delta in *probability* space,
not logit space) made that delta *recoverable*: a pixel only saturates if the actual content
consistently pushes it in the same direction, and since content changes, in practice the
pixel can recover.

CoordConv (v7.28) reopened exactly this vulnerability through a different door: it gave the
coarse grid (4×4/8×8) a *fixed*, clean, *constant* coordinate channel at every step
(content/latent-independent). Once the network learned to use that channel to push the delta
in one direction on weak-signal layers (L5–L7, tiny physical range, where a "know where I am"
shortcut reduces average error more easily than learning real dynamics), that push stops
being recoverable: it is *constant*, not content-dependent, so it pushes in the *same*
direction at *every* rollout step — exactly the unchecked accumulation the v7.6 fix was
designed to prevent, now reintroduced via a different path. The boundary column (where the
coordinate channel takes its most extreme value, −1 or +1) is the exact signature of this
mechanism.

**Fix:** the decoder is reverted to the pre-v7.28 architecture (bilinear + conv, no
CoordConv, no residual smoothing block) — the same architecture the v7.6 fix had already
validated as safe against this specific failure mode. `spatial_curvature_loss` (v7.28) is
*kept* — it's a loss mechanism, not an architecture change, and does not share the risk that
caused this regression; its weights are raised slightly (`W_CURVATURE_AE` 1.0→1.5,
`W_CURVATURE` 1.5→2.0) since it is now the *only* mechanism against blocking.

**Lesson:** the discipline of "revert on evidence of regression, don't insist" (already
applied in v7.11) worked again. The difference this time is that reviewing the user's
reference model allowed pinpointing the cause precisely instead of blindly reverting — the
project had already solved this *exact* problem in v7.4/v7.6, and CoordConv reopened it
unnoticed because the mechanism (a *constant*, content-independent position channel) is
subtle. Any future attempt to give the decoder position information should first pass the
same "can this create a non-recoverable, per-step-constant, content-independent push?" test
before being integrated.

---

## v7.28 — Why the "never change architecture" rule was broken

**Evidence** (real run of v7.27): predictions develop "block"/mosaic-like transitions — hard
boundaries instead of a continuous gradient — visible already at t+10 (not something that
only accumulates over a long rollout) and intensifying with horizon. The real field, by
contrast, is smooth at every step. The user described it precisely: "it lacks smoothness
and intensity in certain zones… the network should be able to say which are the highest and
lowest points and how they blur."

The defect already being present at t+10 (a single rollout step, dominated by Phase
1/decoding reconstruction quality, not accumulated Phase 2.5 drift) is the key signal: this
is *not* a matter of rollout training horizon (those symptoms were already addressed in
v7.20/v7.26/v7.27) — it is a *capacity limitation of the decoder itself*. The decoder (since
v6) only had bilinear upsampling + 3×3 convolutions from a 4×4 grid, with *no* absolute
position signal. Convolutions are translation-equivariant by design — they have no cheap way
to know "I'm at the edge" vs. "I'm at the center" to modulate a smooth, position-dependent
transition. Twelve-plus versions of loss-only tuning (v7.9 to v7.27) improved related
symptoms (level, phase, global shape, orientation, boundary ring) without being able to touch
this specific limitation, because no loss can give a network capacity its architecture
doesn't have.

**Fix (first architecture change since v6):**
1. **CoordConv** (Liu et al., 2018): row/column coordinate channels (normalized to [-1,1])
   concatenated to the input of every spatial convolution in the decoder (at 8×8 and 15×15)
   — an established, low-risk, purely additive technique (new input channels, nothing
   removed). Gives the convolution an explicit "where am I" signal.
2. **Residual smoothing block** at final resolution (15×15):
   `raw_out = raw_out + 0.5*refine(raw_out, coords)` — additional capacity exactly where the
   artifact is visible, without discarding what the original path already learns.
3. **`spatial_curvature_loss`** (new, `W_CURVATURE_AE=1.0` Phase 1, `W_CURVATURE=1.5` Phase
   2.5): compares the discrete 5-point Laplacian (local curvature) of prediction vs. reality
   — a physically smooth field has bounded curvature; a block boundary produces a curvature
   spike reality doesn't have. Complements `masked_gradient_loss` (first differences) by
   comparing second differences, more sensitive specifically to sharp block-like
   discontinuities.

**Validated before integration:** new decoder tested in isolation — correct output shapes,
[0,1] range respected, all parameters (including new ones) receive finite gradients, no
pathological parameter growth (96.6k vs. ~70k previously). Curvature loss tested separately
— ~0 on identical fields, clearly positive comparing a field with injected blocks against
its smooth version, finite gradients.

**Risk and mitigation:** by definition, the highest-risk change in the project so far.
Mitigants: (a) additive, not destructive; (b) the protection mechanisms accumulated since
v7.17–v7.19 (early stopping, best-checkpoint, epoch rollback + LR backoff, NaN guards)
remain intact; (c) if the block artifact persists even with these changes, the conclusion
would be that the limitation runs deeper than upsampling (possibly the 64-dim latent
bottleneck) — valuable diagnostic information in itself, not a loss.

*(As it turned out, this experiment caused a severe regression and was reverted in v7.29 —
see above.)*

---

## v7.27 — The ring is gone (v7.26 fix confirmed); what remains is an orientation drift with a temporal signature pointing at the training horizon

**Confirmations** (real run of v7.26): the L3–L7 long-horizon ring/donut is *gone* (per-pixel
envelope smoothing worked); the period estimator now detects ~629 steps (physically
plausible, the v7.25 bugfix worked); MAPE intact — L1–L8 <1% through t+2000 inclusive (the
user's explicit target for steps beyond 1000 is **met** for 8 of 9 layers; Layer 0 remains at
its ~5% turbulent-decorrelation floor).

**What remains:** the L4 prediction develops a *diagonal* (~45°) blob that intensifies from
t+~500 to t+2000, while reality is a *horizontal* (~0°) band. Progress is notable: it is no
longer circular (v7.25's anisotropy loss did achieve elongation) — only *orientation* now
fails. The temporal signature is the key clue: the drift starts right after t+~320–500, i.e.
immediately after `K_ROLLOUT_STEPS=320`, the maximum horizon the shape/orientation losses
reach during training. Beyond that point the rollout drifts toward the decoder's preferred
attractor — the diagonal, a bias documented since v7.8/v7.17 — with nothing left to correct
it.

**Three changes:**
1. `K_ROLLOUT_STEPS` 320 → 512: extends the horizon over which shape/orientation losses
   shape the dynamics, pushing the drift point further out. Cost ~1.6× per sample, epoch
   time bounded by the batch cap (v7.19), memory unchanged (TBPTT stays in chunks of 24).
2. `W_ANISOTROPY` 1.0 → 2.0: the failing axis (orientation) is exactly what this loss
   penalizes; v7.25's unit tests (finite gradients, monotone optimization) give room to
   raise it.
3. New diagnostic `plot_shape_orientation_tracking`: elongation and orientation curves
   (major-axis angle; 0 = horizontal band) real vs. predicted, step by step over rollouts of
   up to 2000 steps — turns the visual symptom into a number that says exactly *when*
   predicted orientation departs from reality.

**Honest limit:** if the diagonal persisted even with the extended horizon, the remaining
cause would be the decoder's architectural bias (bilinear upsampling, documented since
v7.17) — and the next step would be an architecture change, outside the scope of the
low-risk incremental adjustments this project has preferred.

---

## v7.26 — Mechanistic cause of the long-horizon ring/donut, found and fixed

**Evidence** (real run of v7.25, comparing t+15 vs. t+1000 on the same layers): L3–L7 look
faithful at t+15 but develop a bright ring/donut at the perimeter with a differently-colored
center by t+1000 — a pattern absent at short horizon that *accumulates* with steps. The same
mechanism likely explains why `plot_concentration_flow_field` showed predictions becoming too
*circular* where reality is elongated (a ring naturally produces a circularly-symmetric
divergence flow).

**Mechanistic cause:** the real hot band shifts slightly frame to frame within the training
window — at pixels that were ever "within" the band's reach, the percentile
(`TEXTURE_ENV_LO_PX`/`HI_PX`, v7.20) allows a wide range; at pixels never covered, the range
is narrow. This creates a *hard boundary* exactly where the band's historical reach ends —
invisible in a single step, but since the per-pixel envelope applies at *every* step of a
free rollout, that boundary reinforces/accumulates over hundreds of steps until visible as
the ring.

**Fix:** spatial smoothing of `TEXTURE_ENV_LO_PX`/`HI_PX` with a mask-aware Gaussian blur
(`_smooth_masked_2d`, normalized by the blurred mask so the exterior doesn't contaminate the
boundary) — tested before integration: a hard 0.01→0.10 transition at one pixel becomes a
0.033→0.077→0.096→0.077→0.033 gradient after smoothing. Since smoothing *widens* the
envelope in most cases (averaging with more permissive neighbors), regression risk is low —
a wider envelope restricts less, never more, than the original.

**Also in this version:** `LONG_HORIZONS` (diagnostics script) extended with t+1500/t+2000
(previously capped at t+1000) — explicit user request ("the goal is 1% MAPE beyond 1000
steps"); test data (80% of the set, v7.17 split) has ample margin. An explicit per-layer
pass/fail report against <1% at the longest evaluated horizon is added, with an honest note
on why Layer 0 is the expected exception (genuine turbulent decorrelation, not model
quality) and that evaluating it there requires `plot_turbulence_statistics_check`
(spectrum + histogram), not pixel-wise MAPE.

---

## v7.25 — The flow field (v7.24) exposed an unpenalized orientation mismatch, plus two diagnostic bugs found while testing it

**Main finding** (`plot_concentration_flow_field`, new in v7.24, real run): at short horizon
predicted arrows converge toward the hot spot almost like the real ones in L1–L3 — the model
*did* learn the transport pattern (convergence = concentration), not just the values. But at
long horizon (t+500), L4 (a horizontal *band* in reality) loses that orientation in the
prediction and becomes more radial/circular.

**Cause:** `centroid_spread_consistency_loss` (v7.15) tracks the hot spot with an
*isotropic* radius (a circle) that cannot distinguish "a horizontally-oriented band" from "a
circle of the same size." Nor does `spatial_pattern_correlation_loss` (v7.22, global
pixel-wise correlation) penalize this in a targeted way.

**Fix:** `compute_shape_moments_torch` (new) extends the centroid/spread calculation to the
full second-moment tensor (intensity-weighted covariance ellipse, classic "image moments")
— gives major radius, minor radius, and major-axis orientation. `spatial_anisotropy_loss`
(new, `W_ANISOTROPY=1.0`) compares shape (major/minor ratio) and orientation (as
cos(2θ)/sin(2θ), to avoid the discontinuity of an axis being indistinguishable from itself
rotated 180°) between prediction and reality. Tested before integration: identical shapes
give loss~0, the same shape rotated 90° gives high loss (~4), finite gradients, and SGD
monotonically reduces the loss on random fields. Uses the same per-layer confidence weight
as the centroid loss (v7.18) — layers without a coherent focus shouldn't be forced into an
orientation that is noise there.

**Two diagnostic bugs** found while using v7.24's new tools on real data (documented and
fixed in the diagnostics script):
1. `estimate_dominant_period` returned exactly `min_period` (50) instead of the real
   physical period (~530) — the ACF of a series with a slow trend decays monotonically
   within the search window, and taking the max of a purely-decaying curve gives the
   smallest allowed lag. Fix: detrend by differencing (more robust than a moving average,
   which was found sensitive to window size) + search for the most prominent local peak +
   an optional `expected_period_hint` (informed by this project's earlier DMD analysis,
   ~530 steps) that prioritizes the nearest candidate while still verifying against the
   real data.
2. `plot_concentration_flow_field`'s divergence maps came out almost entirely gray (NaN) —
   the per-pixel rejection threshold (`det(AtA) < eps` with an *absolute* `eps`) depended on
   the layer's physical scale, which varies by orders of magnitude in this project (L0
   ~60–110 vs. L5–L8 ~77.3–77.7), systematically rejecting more pixels in small-range
   layers. Fix: regularization proportional to local scale (`rel_eps * trace`), not an
   absolute threshold — confirmed directly: of 145 possible valid pixels, the old version
   resolved 102, the new one resolved all 145.

---

## v7.24 — The user's vector-field / inter-layer-coupling idea, mapped to what already exists and implemented in its two new pieces

**The idea (verbatim):** "for each layer a vector field to know where the highest mass is
concentrating… analyze the field's movement over time… how lower layers affect upper ones…
how it disperses sideways and whether it's concentrating or diminishing… a kind of PINN…
learn general movements in low/medium fidelity… the first 5 steps as the system's initial
constants."

**Honest mapping to what already exists** (to avoid duplication):
- "concentration point, movement, and dispersion" → centroid/spread tracking (v7.15) + its
  loss (v7.18) are the "point" version of this idea, already training.
- "how lower layers affect upper ones" → the PINN's axial term (`v_z*(T_i - T_{i-1})`)
  already encodes that coupling as a hard equation (Phase 3, reactivated in v7.22).
- "first 5 steps as initial constants" → `SEQ_LEN=5` is exactly that: the context window
  fixing the initial state.
- "learn general movements in low/medium fidelity" → Phase 2.5a (v7.18).

**What is genuinely new, implemented in this version:**
1. **Full vector field** (diagnostics script): `estimate_flow_field_lk` solves, per pixel
   and per layer, the advection equation `dC/dt + v·grad(C) = 0` via local least squares
   (Lucas–Kanade) — gives arrows for where concentration flows and, via the field's
   *divergence* (div<0 = concentrating, div>0 = dispersing), exactly the "how it
   disperses/concentrates" from the idea. `plot_concentration_flow_field` compares real vs.
   predicted at short and long horizons. Deliberately diagnostic-first: lets the transport
   match be evaluated empirically before betting training on a flow loss (which would
   require differentiable warping — higher risk).
2. **`interlayer_coupling_loss`** (`W_INTERLAYER=1.0`, trains in Phase 2.5): Pearson
   correlation between the deviation patterns of each pair of adjacent layers, penalizing
   the predicted coupling profile differing from the real one — the trainable/statistical
   counterpart to the axial coupling the PINN imposes as a hard equation. Inside the
   warm-up ramp with the usual protections.

**Also:** `W_GRAD_ROLLOUT` 0.3 → 0.5 — the real v7.23 run reduced the boundary ring without
a MAPE regression (reactivation was correct), but the interior of L4–L6 at long horizon
became "speckled" (small blobs where reality is smooth); more gradient weight pushes toward
real smoothness.

**v7.23 reference results:** targets intact — L1–L8 ≤0.84% at all horizons, L0 4.85–5.49% at
t+200..t+1000 (slightly better than v7.22 at long horizon).

---

## v7.23 — The v7.22 run is the project's best so far; two small adjustments for what remains

**v7.22 results** (real run): L1–L8 all ≤0.64% MAPE at every horizon through t+1000 (<1%
target met with margin); L0 at 5.31–5.55% at long horizon (down from 9–11%) and <5% through
t+200; the spatial pattern returned to place (centered bulge in L1–L4, the v7.21 shifted band
disappeared — pattern correlation worked exactly as designed); a 6000-step continuous
simulation, stable and unbiased.

**Two adjustments:**
1. **Boundary ring** in L4–L5 at long horizon (bright perimeter + slight interior
   fragmentation). Mechanism: boundary pixels have genuinely wide real ranges (the per-pixel
   envelope can't tighten them without harming legitimate cases), and std-renorm compensates
   for edge brightness by flattening the interior. Fix: `W_GRAD_ROLLOUT` 0.0 → 0.3 — the
   spatial gradient loss penalizes exactly the spurious boundary gradients. It was disabled
   in v7.8 (weight 1.0, regression), but the context has changed: pattern correlation
   (v7.22) now anchors global shape, and checkpoint/early-stop/probe/rollback (v7.17–v7.19)
   protect against regressions. Deliberately small weight.
2. `W_VAR_CEIL` 5 → 8, `VAR_CEIL_RATIO` 1.3 → 1.2: predicted oscillation amplitude still
   somewhat above real at short horizon.

---

## v7.22 — MAPE was the project's best yet, but the spatial pattern was wrong: root cause identified and fixed

**Symptom** (real run of v7.21): numerically, the project's best run yet — L1–L8 all
≤1.01% MAPE at every horizon through t+1000, L1 finally under 1% almost everywhere. But
visually, in L2–L5 the prediction was an elongated band shifted to the left where reality is
a centered, smooth bulge — the low average error hid that the *pattern* was in the wrong
place.

**Root cause** (confirmed by comparing against an older model the user uploaded, a v7.10
whose t+15 fields were visually faithful): that model had *neither* of the two new spatial
losses from v7.15+. Of the two, `spatial_spectral_loss` has a precise mathematical flaw: it
compares 2D FFT *magnitude*, which is *translation-invariant* — the model can put the
correct spectral energy in a shifted band and the loss is equally satisfied. Combined with
the v7.21 spread sub-weight increase, the optimizer found that a "shifted elongated band"
satisfies energy+spread even with the pattern mislocated.

**Fix:**
1. `W_SPATIAL_SPECTRAL`: 0.7 → 0.0 (disabled by default, code intact — same pattern as
   `W_GRAD_ROLLOUT`/`W_SPATIAL_VAR` in v7.8).
2. New `spatial_pattern_correlation_loss` (`W_SPATIAL_CORR=4.0`): 1 − Pearson correlation
   pixel-by-pixel between predicted and real *deviation* fields, inside the mask.
   Position-sensitive (a shifted pattern gives low correlation) — exactly "the same pattern
   in the same place," the spatial analog of `temporal_shape_loss` (the temporal loss that
   has worked best throughout the project). Operating on deviations, it doesn't fight with
   level (trend renorm).
3. `CENTROID_SPREAD_SUBWEIGHT`: 0.6 → 0.3 (reverts the v7.21 increase, which contributed to
   elongation).

**Phase 3 (PINN) reactivated**, at the user's request, with safeguards the project's
history demands (v7.4: "the residual never fit well and degraded rollouts"; v6.1: "high
W_PINN dominated the gradient"):
- `W_PINN` 0.5 → 0.2, `EPOCHS_PINN_FINETUNE` 10 → 6, full warm-up, physics calibrated on
  ground truth and frozen, NaN guard in the training step (previously missing).
- **Automatic reversion** (new): model snapshotted before Phase 3; afterward, the
  stability probe + quick Val MAPE run — if the probe diverges or Val MAPE worsens by more
  than `PINN_ACCEPT_TOLERANCE` (10% relative), the model is *restored* to the snapshot and
  the physics tuning is discarded for that run (with a clear log message). Phase 3 can no
  longer leave the model worse than it found it.

---

## v7.21 — The v7.20 run confirmed both fixes; three targeted adjustments for what remains

**Confirmations** (real run, compared side-by-side against v7.19): level bias of the
continuous simulation *eliminated* (prediction oscillates around reality on all 9 layers;
previously always ~0.07 below); DMD sawtooth: peaks of |d(trend)/dt| from ~0.2–0.4 down to
~0.035 (~10× less); temporal jitter from ~6× real to ~1.5–2×; "plume" artifact (top edge,
L0–L4 at t+1000) *gone* — per-pixel envelope worked (one residual bright pixel remains at
L8's extreme boundary, legitimately wide-ranged in training, not a bug); L1 improved
throughout (max 1.18% vs. 2.48%); L0 at t+1000: 11.2%→9.0%.

**Three targeted adjustments:**
1. `variance_ceiling_loss` (new, `W_VAR_CEIL=5`, ratio 1.3): symmetric counterpart to the
   variance floor (v6.1). The floor prevents collapse to constant but nothing penalized
   *excess* — and the residual jitter (~1.5–2×) is exactly that.
2. `CENTROID_SPREAD_SUBWEIGHT`: 0.3 → 0.6. The hot spot's *location* is already tracked
   well, but the prediction is *too concentrated* (tight/diagonal blob in L3–L5 where
   reality is a wide, diffuse band) — a *spread* mismatch, not position.
3. `EPOCHS_ROLLOUT_FINETUNE`: 20 → 30. With training already stable and per-epoch cost
   bounded (v7.19), more Phase 2.5b epochs attack the one remaining numeric target (L1 at
   1.16–1.18% at t+500+, just over the 1% goal).

**Note on L0 at long horizon (~9%):** consistent with the genuine turbulent-decorrelation
floor — the correct evaluation there is the *statistical* one
(`plot_turbulence_statistics_check`: spectrum + histogram), not pixel-wise MAPE.

---

## v7.20 — Two problems with an identified cause in the (then-best) v7.19 run, both fixed and validated before integration

**Problem 1: the rolling DMD trend injected level bias and jitter.**
Evidence (three signals pointing to the same cause): (a) in the continuous simulation (8000
steps), predictions were systematically ~0.07 *below* reality on all 9 layers; (b) predicted
temporal std came out ~6× the real value at horizon 15 — high-frequency jitter; (c) the
"fingerprint": peaks of |d(trend)/dt| occurred every ~25 steps == `DMD_RESEED_EVERY=25` — the
resembling sawtooth. Mechanism: the DMD eigenvalues are stabilized to |λ|≤1 (v7), so within
each 25-step segment the forecast *decays* toward its fixed point = the training-window
average — which, with the 10/10/80 split, is only the *first* 10% of the series and is
stale relative to the real val/test level. Each segment drags the level toward the past, the
reseed snaps it back (sawtooth), and since `apply_trend_renorm` pins the predicted field to
this trend, both errors are injected directly into the final prediction.
**Fix** (`simulate_dmd_trend_rolling`, two new parameters):
- `local_recenter=True`: each segment is centered on the *local* average of its own seed
  (recent real data) instead of the global training average. The linear dynamics A are
  translation-invariant, so this is mathematically consistent — it only changes the decay's
  fixed point.
- `crossfade_steps=5`: linear blend at each reseed boundary, eliminating the sawtooth.
Validated on synthetic data (horizon 800): jitter 0.107 → 0.092 (real: 0.090 — now nearly
identical), level bias +0.0094 → −0.0021 (~4.5× smaller), trend MAE also improves (0.0906 →
0.0859).

**Problem 2: a fixed-location "plume" artifact** (top edge in predicted L0–L4 at t+1000;
corner blobs in L8 at t+15). The whole-layer scalar texture envelope (v7.9) bounds the
*magnitude* of deviations but not their *location* — a +0.05 plume at the top edge is within
the allowed scalar range even though reality never has such deviations at those pixels.
**Fix:** per-pixel envelope (`apply_texture_envelope_px`, percentiles 0.5/99.5 of train *per
pixel*, with a 1.25× multiplicative margin + an additive margin of 0.25×typical layer std to
compensate for percentile sampling noise with only ~10% of the training data). This is
v7.11's idea, previously reverted for a broadcasting bug — integrated this time only after
passing explicit unit tests for shape/broadcasting, near-no-op behavior on real training
frames, and effective clipping of an artificially injected plume.

**Unchanged:** architecture (zero changes), losses/weights (v7.18/v7.19), the 10/10/80
split, cost control and safety nets (v7.19).

---

## v7.19 — Preventive hardening after a training failure reported in the v7.18 real run

**Honesty note:** the failure log did not arrive in readable form, so — unlike previous
addenda that fix a *confirmed* cause — this version hardens training against *all* plausible
v7.18 failure modes, most-likely first. Much more informative logging is added so the next
run's log will identify the actual cause unambiguously.

**Failure mode #1 (most likely): explosive Phase 2.5a cost.** With real data, low/medium
fidelity have thousands of steps → thousands of overlapping rollout windows, each epoch
traversing all of them, each batch training rollouts of up to 320 steps with backward — a
per-epoch cost that can become impractical (hours/days) or exhaust memory. Two fixes:
1. `ROLLOUT_WINDOW_STRIDE=8` (new, `RolloutDatasetMF`): step-by-step overlapping windows
   share ~99.7% of content at k_steps=320 — sampling every 8 steps removes near-pure
   redundancy, cutting window count ~8× without losing temporal coverage. Low/medium
   fidelity only; high fidelity keeps stride=1 (no window from the scarce 10% is dropped).
2. `MAX_ROLLOUT_BATCHES_PER_EPOCH=30`: a hard per-fidelity, per-epoch cap in 2.5a (and also
   in 2.5b as a safety belt) — guarantees bounded epoch time regardless of dataset size.
`make_loaders` now prints the resulting window count after stride/cap, to estimate per-epoch
cost before training starts.

**Failure mode #2: numerical instability (NaN/Inf) outside Phase 2.5.** v7.18 only protected
Phase 2.5 (per-chunk guard). Now Phase 1 (`train_step_autoencoder`) and Phase 2
(`train_step_dynamics`) have the same guard: a batch with non-finite loss is discarded
without touching weights.

**Failure mode #3: an entire epoch corrupted** (the per-batch guard isn't enough). New
epoch-level rollback + LR backoff mechanism in 2.5a/2.5b: a snapshot is saved at the start of
each epoch; if the fraction of non-finite batches ≥ `NONFINITE_EPOCH_FRACTION` (50%), the
epoch is entirely discarded (rollback to the snapshot), the LR is halved, and the epoch is
retried. After `MAX_LR_BACKOFFS` (3) consecutive reductions without a healthy epoch, the
phase stops cleanly, keeping the last good state. Every 2.5a/2.5b epoch log line now also
reports how many batches were discarded for being non-finite.

**Unchanged:** architecture, losses/weights (v7.18), the 10/10/80 split (v7.17), the rest of
the pipeline. Look for "[!]", "backoff", or "discarded" in the log if the next run fails
again.

---

## v7.18 — Phase 2.5 exploded to NaN in the v7.17 real run, plus three explicit user requests

**Critical bug found and fixed:** the v7.17 real run showed the centroid loss starting at
~8.5 (epoch 1) and climbing to ~13.8 (epoch 10), with anomalous intermediate values, until
everything became NaN by epoch 18 — the last 3 epochs of Phase 2.5 were entirely wasted
(early stopping saved the run only because a good checkpoint had already been saved before
this happened).

**Diagnosis:** `centroid_spread_consistency_loss` treated all 9 layers equally, but not all
have a well-defined "hot spot" — the user noted directly in the images that one layer shows
ambiguous/dispersed behavior (multiple foci that shift place frame to frame, no single stable
hot spot), while others have much clearer motion and dissipation. For the ambiguous layer,
the "real centroid" the loss tried to match is itself *noisy* — it jumps a lot step to step
even for a physically valid field — and chasing that noise with a fixed weight destabilized
training.

**Fix (two layers of protection):**
1. **`CENTROID_LAYER_WEIGHT`** (new, computed from the data itself): the variance of the
   real centroid's *step* (frame to frame) is measured per layer over the training window —
   layers where the real centroid jumps a lot (no general focal point, ambiguous behavior)
   get a weight *near zero* in the loss; layers with stable, clear movement/dissipation get
   full weight. `W_CENTROID` is also lowered as extra caution (1.5 → 1.0), though the real
   fix is the per-layer weight, not this scalar.
2. **NaN/Inf guard** (new, in `train_step_rollout_consistency`): if a chunk's loss stops
   being finite, that batch is discarded without touching the weights or optimizer state
   (instead of propagating the NaN through `.backward()`/`.step()`, which would corrupt the
   model for the rest of training). A general safety net, not only for this specific cause.

**New — Phase 2.5a: low/medium-fidelity rollout pretraining** (explicit user request:
"low-fidelity models should also learn how each layer's physics behaves, so it can learn it
more easily afterward with high-fidelity data"). Adds `pretrain_rollout_consistency` (Phase
2.5a): exactly the same training mechanism (TBPTT, k_steps curriculum, same losses) but on
abundant low+medium fidelity — Phase 2.5b (formerly "Phase 2.5," renamed for clarity) then
starts from an already-reasonable rollout dynamic, with much less work left to do on the
scarce 10% high-fidelity data. Same pretrain/finetune pattern already used by Phase 1 (1a/1b)
and Phase 2 (2a/2b) — no architecture change.

**New in diagnostics** (v7.18 diagnostics script):
- `plot_density_peaks_check`: counts local density maxima ("dense points") per layer/horizon,
  real vs. predicted — answers directly "how many dense points are there" and whether the
  model captures a single coherent focus or a dispersed, turbulence-like pattern, per layer.
- `plot_full_cycle_replication_check`: estimates the dominant oscillation period by
  autocorrelation directly from real data (not assumed), cuts `n_cycles` full cycles spaced
  one period apart, and runs a full free rollout of one complete cycle from each — explicit
  user request ("cut cycles… to see if the model can replicate them").

---

## v7.17 — Explicit split constraint + two findings from the real v7.16 run

**Split change** (explicit user request — "only 10% of high-fidelity training data may be
used for everything, another 10% for validation, and 80% for test"): the separate
"statistics" window that existed since v7.14 (previously 50%, larger than the 10% used for
gradient training) is removed. The *same* 10% of high fidelity is now used for *everything*:
fitting the DMD trend, texture envelope, centroid/spread model, *and* the gradient-based
fine-tuning of Phases 1b/2b/2.5/3. `HIGH_FIDELITY_VAL_FRAC` goes from 0.25 to 0.10; test is
now ~80% (previously ~25%) — a much larger, more robust evaluation set.

Real consequence compensated in code: with 5× less data for the DMD, the previous
`DMD_EMBED_DIM` heuristic (`len//6`) would have given an embed_dim of ~166 — too short
relative to the real ~530-step oscillation period, likely insufficient to capture it (per
Takens, the embedding needs to be comparable to the period). The heuristic changed to
`len*0.6` to get back closer to the ~600 steps used in earlier versions despite having a
tenth of the data.

**Two findings from the real v7.16 run** (symmetry fix already validated: Phase 1b Val MAPE
back to ~8.4%, best ever) **and their fixes:**
1. **The Phase 2.5 "best checkpoint" metric was misaligned with that phase's goal.** The log
   showed "quick Val MAPE" (measured at only 15 steps) worsening *monotonically* throughout
   Phase 2.5 (0.51% → 0.65%), making early stopping always revert to epoch 2 — not because
   that epoch was truly best, but because a 15-step metric is blind to the long-horizon
   improvement Phase 2.5 exists to provide. Fix: `CHECKPOINT_EVAL_HORIZON=200` (new) — both
   `val_ds` and `quick_rollout_val_mape` now evaluate at this much longer horizon.
2. **New symptom: long-horizon texture over-fitting.** `plot_texture_directionality_check`
   in the real run showed several layers (L0–L2) with ratio >1.4–1.9 at t+500/t+1000 — the
   model now generates *more* spatial structure than reality, the opposite of the
   blob-collapse fixed in v7.9/v7.15. Two conservative responses: `W_SPATIAL_SPECTRAL`
   lowered again (1.0 → 0.7), and a new warm-up ramp (`SPATIAL_LOSS_WARMUP_EPOCHS=8`) for
   `W_SPATIAL_SPECTRAL`/`W_CENTROID` — both start at 0 and ramp up linearly.

**Identified but *not* resolved in this version:** `plot_mape_evolution_multi_horizon` showed
a diagonal/band pattern in predicted Layer 0 that looked almost identical from t+10 through
t+1000 — a signature the v7.8 addendum already identified as a *fixed* architectural bias
(content-independent), not genuine instability. `check_bifurcation_is_static`'s default
layer changed to `layer=0` in the diagnostics script to confirm this directly.

**Also new:** `plot_turbulence_statistics_check` (diagnostics script) — the statistical
(spectrum + histogram) evaluation for Layer 0 that v7.9/v7.14 had promised but never
implemented.

---

## v7.16 — Real regression found in v7.15 (real data, `HIGH_FIDELITY_TRAIN_FRAC=0.10`) and its fix

The first real v7.15 run confirmed something good *and* something bad simultaneously:

**Good news (confirmed):** the log printed the 8 valid D4 dihedral-group symmetries for the
real core mask — the 4-loop reactor genuinely has that geometric symmetry.

**Bad news:** applying symmetry augmentation to the single-frame autoencoder (Phase 1)
made things worse, on two independent signals: (1) Phase 1b Val MAPE landed at ~16.7–18.5%,
much worse than v7.14's ~9.7–10.7% *despite* training for twice as many epochs; (2)
`plot_spatial_error` at t+1000 showed a *new* cross-shaped bright artifact (vertical +
horizontal through the grid center) in several layers, not matching the real hot spot's
location.

**Diagnosis:** `ConvFeatureExtractor`/`BoundedDecoder` are an ordinary, non-rotation-
equivariant CNN with a small latent bottleneck (`LATENT_DIM=64`). Phase 1's autoencoder gives
the decoder no signal of "which of the 8 orientations is this" — it must infer it purely
from the 64-number latent vector, jointly with the real physical content. Without an
explicit pose signal, the optimizer's cheap path is a kind of "average" over the 8
orientations — and the pattern that survives averaging a directional gradient under the four
90° rotations is precisely a cross through the center. This differs from Phase 2/2.5, where
the same transformation is applied to context *and* future together, so "predict the next
frame in the same orientation as the context" remains well-defined — but since the encoder
is shared (and already confused from Phase 1), the problem propagates anyway.

**Fix (minimal, surgical, zero architecture changes):**
1. `USE_SYMMETRY_AUGMENTATION` defaults to `False`. The full mechanism stays intact and is
   still computed/printed at data load time (real, useful physical information) — simply not
   used by default until there's a safe way to re-enable it.
2. `FrameDatasetMF` (Phase 1, single frames) *never* applies symmetry now, regardless of the
   flag — the call was removed from that class entirely. Small noise jitter is kept (much
   smaller effect, no evidence it caused harm). `NextFrameDatasetMF`/`RolloutDatasetMF`
   still respect the global flag if re-enabled.
3. `W_SPATIAL_SPECTRAL_AE`: 1.0 → 0.3 (secondary, unconfirmed but plausible hypothesis: the
   2D spatial FFT loss includes the mask's irregular boundary, introducing high-frequency
   "ringing" unrelated to the interior texture).
4. `W_SPATIAL_SPECTRAL` (Phase 2.5): 2.0 → 1.0, same design-symmetry caution.

`W_CENTROID`/`CENTROID_SPREAD_SUBWEIGHT` are *not* touched — this piece worked according to
evidence (`plot_centroid_tracking` showed L1 centroid drift oscillating within 0.05–0.85
pixels over 1000 full steps, vs. 1.5–2 pixels with clearly divergent trajectories without
this loss).

**Idea for a future version (not implemented, higher-risk architecture change):** to recover
symmetry augmentation for Phase 1 safely, give the decoder an explicit *pose* signal (e.g. an
8-way one-hot concatenated alongside the fidelity embedding) indicating which transform was
applied — analogous to how fidelity is already provided.

---

## v7.15 — Motivation and additive fixes for the `HIGH_FIDELITY_TRAIN_FRAC=0.10` regime

With only 999 high-fidelity steps for fine-tuning, two real problems appeared that earlier
versions (with 50% of the data) didn't show, or showed much more mildly:
1. **Layer 0** worsened: it already failed the 5% MAPE target from t+100 onward (11.76%,
   previously only failing from t+750). The Phase 1b autoencoder had not fully converged by
   epoch 15.
2. **L4–L8 collapse to a near-uniform "blob"** from t+50 onward, losing the directional
   gradient the real field retains. Invisible in aggregate MAPE because these layers' physical
   range is tiny (~0.2–0.5 units): predicting the average instead of the texture still gives a
   low MAPE%, even though 100% of the useful signal is lost.

**Root diagnosis:** with so little fine-tuning, none of the existing guarantees
(`apply_texture_std_renorm`, `apply_texture_envelope`, `apply_trend_renorm`) protect the
spatial pattern's *shape/location* — they are all per-layer scalars (amplitude, range,
level), saying nothing about *whether* the gradient is where it should be.

**Changes in this version (all additive — zero architecture changes):**
1. **Geometric-symmetry data augmentation** (`find_valid_symmetries`, new): programmatically
   tests the 8 D4 dihedral-group transforms against the fixed core mask, using only those
   that leave it *exactly* unchanged (verified, not assumed). Applying the same transform to
   all 9 layers and all frames of a window/sequence at once produces an equally valid field.
   Applied to low, medium, *and* high fidelity; never to val/test/probe.
2. **Small Gaussian noise jitter** (`maybe_jitter_noise`), context only, never the loss
   target.
3. **`spatial_spectral_loss`** (new): analog of `spectral_shape_loss` (v6.1, temporal axis)
   applied to the spatial (H,W) plane of a single frame — compares 2D FFT magnitude of the
   deviation from the layer average. A uniform blob has ~0 spectral energy at non-zero
   frequencies; this loss gives well-conditioned gradient to recover it.
4. **`centroid_spread_consistency_loss`** (new): differentiably tracks the intensity-weighted
   centroid and spread of each layer — same definition already used diagnostically
   (`compute_centroid_and_spread`, v7.12), now in PyTorch so it can be used as a loss.
5. **`fit_centroid_dynamics`** (new): reuses the same Hankel-DMD machinery already used for
   trend forecasting, applied to the (row-centroid, col-centroid, spread) time series per
   layer. Produces `CENTROID_FORECAST_VAL`/`CENTROID_FORECAST_TEST` for future
   diagnostics/extension; deliberately *not* injected as a hard rollout correction (would
   need differentiable spatial warping, a higher-risk architecture change).
6. **Early stopping + best checkpoint** in Phase 1b and Phase 2.5: `EPOCHS_AE_FINETUNE`
   (15→40) and `EPOCHS_ROLLOUT_FINETUNE` (10→20) raised, always keeping the best checkpoint
   and stopping early on no improvement or detected divergence.

All new weights start conservative by design (lesson from v7.7: a new spatial loss with high
weight interacted badly with existing post-decode guarantees and made things worse).

---

## v5 → v7.14 — Full history summary (unedited addenda preserved in earlier project snapshots)

**What changes relative to v5** (`LatentDiffusionDynamics`, a full 50-step ancestral DDPM)
**and why:** v5 used a 50-step ancestral DDPM to model the `z_t → z_{t+1}` transition. v6
separates roles (deterministic causal Transformer + generative Rectified Flow residual,
following G-LED, Kim et al. 2024) and adds Phase 2.5 (rollout training, exposure-bias fix)
and optional physics-guided inference (inspired by Raissi et al.'s PINN framework). v7 adds
trend conditioning via DMD (Brunton et al., HAVOK-style) because `SEQ_LEN=5` cannot, by
construction, recover the phase of a ~500-step oscillation. v7.2–v7.10 add
by-construction guarantees (level renorm, texture envelope, spatial std renorm) that make
spatial saturation/bifurcation mathematically impossible at any horizon. v7.14 introduces the
two-level split enabling fine-tuning with a small high-fidelity fraction.

*(The original, line-by-line addenda for each v5–v7.14 decision were preserved in this
project's earlier working files; the summary above captures the substance carried forward
into every later version.)*
