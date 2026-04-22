# GaussianNLL Training Fixes for S4D Parameter Estimation

**Date:** 2026-04-18  
**Affected files:**
- `src/BNSReg/losses/beta_nll.py` *(new)*
- `src/BNSReg/tasks/parameter_estimation/model_s4d_gaussnll.py`
- `src/BNSReg/tasks/parameter_estimation/model_linoss_gaussnll.py`
- `src/BNSReg/models/s4d.py`
- `src/BNSReg/core/config.py`
- All `configs/parameter_estimation/**/*.yaml` using `LitModelS4DGaussianNLLLoss` or `LitModelLinOSSGaussianNLLLoss`

---

## Diagnosis

Training with `GaussianNLLLoss` on the S4D architecture exhibited a characteristic failure mode: the loss would drop sharply in epoch 0 and then flatline indefinitely. WandB grad norm logs from a representative run confirmed three distinct problems:

```
decoder.weight grad norm:    ~7–30   (mean 7.0)
SSM kernel (log_dt) grad:    ~0.006
SSM kernel (log_A_real) grad: ~0.006
ratio:                        ~1000×
```

The model had converged to predicting the **unconditional distribution** of the target (e.g. chirp mass):
`val/gaussnll ≈ −0.76 ≈ −H(p_marginal)`, confirming the model was outputting the marginal
mean ± variance of the training targets regardless of the input waveform.

The three root causes that compounded each other are described below.

---

## Fix 1: β-NLL Loss (primary fix)

### The problem

Standard `GaussianNLLLoss` has a degenerate local minimum described formally in:

> Seitzer et al. (2022), *"On the Pitfalls of Heteroscedastic Uncertainty Estimation
> with Probabilistic Neural Networks."* https://arxiv.org/abs/2203.09168

The per-element NLL is:

```
L_NLL = 0.5 * [log(var) + (mean − y)² / var]
```

The gradient with respect to `mean` is:

```
∂L_NLL/∂mean = (mean − y) / var
```

A model can drive this gradient to zero without learning the conditional mean simply by
inflating `var`. Once `var` is large, `(mean − y)² / var ≈ 0`, the loss floor is
`0.5 * log(var)` which is minimised at the variance of the target marginal distribution.
The decoder — directly adjacent to the loss — finds this solution within the first epoch
(typically first few hundred steps). Once there, the SSM produces features that are
irrelevant to the loss, so gradients flowing back into the SSM are ~1000× smaller than
decoder gradients. The model is trapped.

### The fix

β-NLL weights the NLL by `stop_gradient(var)^β`:

```
L_β = sg(var)^β · L_NLL

∂L_β/∂mean = sg(var)^(β−1) · (mean − y)
```

| β   | mean gradient           | behaviour |
|-----|-------------------------|-----------|
| 0   | `(mean−y) / var`        | standard NLL — degenerate |
| 0.5 | `(mean−y) / std`        | recommended default; gradient decays slowly |
| 1   | `(mean−y)`              | pure MSE for mean; variance fully decoupled |

With β > 0, the model **cannot suppress mean gradients by inflating variance**. The
decoder is forced to learn the conditional mean, which propagates non-trivial gradients
back through the SSM at every step.

### Implementation

`src/BNSReg/losses/beta_nll.py` — `BetaNLLLoss(beta, reduction)`:

```python
nll = 0.5 * (torch.log(var) + (mean - target)**2 / var)   # standard NLL per element
if beta > 0:
    nll = nll * var.detach().pow(beta)                      # weight by sg(var)^β
loss = nll.mean()
```

`BetaNLLLoss` replaces `torch.nn.GaussianNLLLoss` in both `LitModelS4DGaussianNLLLoss`
and `LitModelLinOSSGaussianNLLLoss`. The `beta_nll` hyperparameter (default `0.5`) is
exposed as a YAML init arg on both tasks.

**Note on interpretation:** the β-NLL value is not directly comparable to standard NLL
values from previous runs. Use `val/mse/out_0` as the primary convergence metric.

---

## Fix 2: S4D Parameter-Group Optimizer

### The problem

`S4DKernel.register()` stamps a `_optim` attribute on the SSM parameters `log_dt`,
`log_A_real`, and `A_imag`:

```python
setattr(param, '_optim', {'weight_decay': 0.0})          # when model_cfg.lr is None
setattr(param, '_optim', {'weight_decay': 0.0, 'lr': x}) # when model_cfg.lr is set
```

This is the S4 paper's mechanism for giving SSM parameters a higher learning rate and
zero weight decay. However, `LightningCLI`'s automatic optimizer injection (from the
`optimizer:` YAML block) creates a single flat param group — `_optim` attrs are never
read, and SSM parameters get the same LR as everything else.

For the curriculum config this meant all params trained at `lr = 1e-5` (old value),
scaling down to `1e-7` during warmup. S4D SSM parameters typically need `lr ≈ 1e-3` to
learn effectively.

### The fix

`LitModelS4DGaussianNLLLoss` now defines `configure_optimizers` directly, using the S4
parameter-group pattern:

```python
# params without _optim → default group (base_lr, weight_decay)
# params with _optim    → one group per unique _optim dict
#   if _optim contains 'lr' key → use that LR (set by model_cfg.lr)
#   otherwise                   → fall back to base_lr
```

This replaces the top-level `optimizer:` / `lr_scheduler:` YAML blocks (which have been
removed from all configs). Optimizer and scheduler hyperparameters are now part of
`model.init_args`:

```yaml
model:
  init_args:
    model_cfg:
      lr: 1.0e-3        # SSM-specific LR (activates _optim lr key)
    base_lr: 1.0e-4     # LR for all non-SSM parameters
    warmup_epochs: 15
    T_0: 16
    T_mult: 1
    eta_min: 1.0e-7
    warmup_start_factor: 0.01
```

`WarmupCosineAnnealingWarmRestarts` receives an optimizer with two param groups and sees
`base_lrs = [1e-4, 1e-3]`, scaling each group independently through warmup and cosine
cycles. When `model_cfg.lr` is `null` (default), SSM parameters fall back to `base_lr`
and behaviour is backward-compatible with the old single-group setup.

**YAML migration:** all existing `configs/parameter_estimation/**/*.yaml` that used
`LitModelS4DGaussianNLLLoss` or `LitModelLinOSSGaussianNLLLoss` were updated to remove
top-level `optimizer:` / `lr_scheduler:` blocks and add the six optimizer init args
under `model.init_args`. User configs with plain `CosineAnnealingWarmRestarts` (no
warmup) were migrated with `warmup_epochs: 0`, which is equivalent.

---

## Fix 3: SSM Diagnostic Logging

### The problem

Without visibility into SSM kernel parameters during training, kernel collapse goes
undetected. The A matrix real parts and dt step sizes are the two quantities that most
directly indicate whether the SSM is learning useful temporal structure.

### The fix

`on_after_backward` in `LitModelS4DGaussianNLLLoss` now logs per-layer:

| metric | meaning |
|--------|---------|
| `ssm/A_real_mean/…` | mean of `−exp(log_A_real)` — how negative A's real parts are |
| `ssm/A_real_max/…`  | most-negative A real part in that layer |
| `ssm/dt_mean/…`     | mean step size `exp(log_dt)` |
| `ssm/dt_max/…`      | largest step size in that layer |

**Interpreting these logs:**

- `A_real` should become *less* negative over training as the model learns to retain
  long-range memory. If it stays near `−0.5` (the init value) for many epochs, the SSM
  kernel is not being updated.
- `dt_max` should ideally remain below `≈ 1 / (|A_real| · L)` for the kernel to survive
  the full sequence length `L`. For `L = 1024` and `|A_real| = 0.5`, this threshold is
  `dt_max ≈ 0.002`. The default `dt_max = 0.1` places roughly half of all channels above
  this threshold at initialisation, causing those channels' kernels to decay to zero
  within the first ~20 steps of a 1024-step window (see note below).

---

## Fix 4: Optional Input Normalisation

`S4Model` accepts `input_norm: bool = False` (also exposed via `S4DModelConfig`). When
enabled, a `nn.InstanceNorm1d(d_input, affine=True)` is applied to the input along the
time axis before the encoder, normalising per-channel amplitude within each sample.

For whitened GW data this is not always necessary (whitening already targets unit PSD),
but it is useful when the amplitude scale varies across the SNR range of a curriculum
(SNR 5–50 spans a 10× amplitude range).

---

## Fix 5: dt Range Tuned to Sequence Length (root cause of persistent collapse)

### The problem

Even after β-NLL was applied, training still flatlined. WandB diagnostics on the β-NLL
run showed:

```
ssm/dt_mean (layer 0):   0.023   (not moving from init over 7 epochs)
ssm/A_real_mean:        −0.498   (barely moving from init −0.5)
```

The gradient of the loss w.r.t. `log_dt` is approximately:

```
∂L/∂log_dt  ∝  Σ_l  l · K[l] · (∂L/∂y_l)
```

where `K[l] = exp(A_real · dt · l)`. With `dt = 0.023` and `A_real = −0.5`:

```
K[200] = exp(−0.5 × 0.023 × 200) = exp(−2.3) ≈ 0.10
K[500] = exp(−0.5 × 0.023 × 500) = exp(−5.75) ≈ 0.003
K[1024] ≈ 0
```

**The kernel is dead for 80% of the 1024-step window.** The gradient ∂L/∂log_dt receives
no signal from lags > ~200. This is a chicken-and-egg trap: to learn that dt should be
smaller (to capture long-range dependencies), the model needs gradient from long lags —
but it only gets gradient from short lags because dt is already too large. The model
cannot escape without a better initialisation.

β-NLL fixes variance inflation at the loss but cannot fix a Jacobian that is zero inside
the network.

### The fix

`dt_max` must be set so that **all channels have non-collapsed kernels at lag L at
initialisation**. The theoretical constraint (treating A_real as fixed at init value −0.5)
is:

```
exp(−0.5 · dt_max · L) ≥ threshold
→  dt_max ≤ −2 · log(threshold) / L
```

For L = 1024 and threshold = 0.3:  `dt_max ≤ 0.0023`

**However, this theoretical bound is too conservative in practice.** Because A_real is
also a learned parameter, the model co-adapts A_real and dt — converging to larger dt
with correspondingly less-negative A_real. Empirical evidence from a well-trained
checkpoint on the same SNR 15-30 dataset (epoch 1096, `SAVED_RESULTS/S4D_GAUSSNLL_SNR_15_30_Mc_q_20260312/`)
shows the model converged to:

```
dt_mean ≈ 0.058   dt_max ≈ 0.22   A_real_mean ≈ −0.78
```

These values violate the theoretical bound yet produce a well-functioning model.
The theoretical constraint over-constrains the initialisation range.

**Recommended empirical settings** (derived from the reference checkpoint):

```yaml
dt_min: 1.0e-3     # matches reference ckpt min of 3.5e-3 (within one order of magnitude)
dt_max: 3.0e-1     # matches reference ckpt max of 0.22
```

The previously recommended values `dt_min: 1.0e-4, dt_max: 2.0e-3` constrain the model
to a range 100× smaller than where it actually converges, starving the SSM of temporal
resolution during training.

---

## Recommended Configuration for Curriculum Training

The curriculum config `chirp_mass_1-5s_d256_s32_l4.yaml` incorporates all fixes:

```yaml
model_cfg:
  lr: 1.0e-3          # SSM group LR — 10× above base_lr
  dt_min: 1.0e-3      # empirically derived from reference checkpoint
  dt_max: 1           # empirically derived from reference checkpoint
beta_nll: 0.5         # β-NLL: prevents variance inflation collapse
lambda_spread: 0.1    # underdispersion penalty (softplus, added 2026-04-22)
base_lr: 1.0e-4       # non-SSM group LR
warmup_epochs: 15
T_0: 16
```

To ablate individual fixes:
- `beta_nll: 0.0` → reverts to standard NLL
- `model_cfg.lr: null` → all params use `base_lr` (no SSM LR boost)
- `dt_max: 0.1` → reverts to default (collapsed channels for L ≫ 200)

---

## Reference: dt_max vs Sequence Length

> **Note:** The table below is based on the theoretical constraint
> `exp(−0.5 · dt_max · L) ≥ threshold` treating A_real as fixed.
> Empirically, the model converges to dt ≈ 0.06–0.22 regardless of sequence length
> because A_real co-adapts. Use `dt_min: 1e-3, dt_max: 3e-1` as a robust default
> across all sequence lengths rather than the table values.

| L    | window @ 256 Hz | theoretical dt_max (30%) | empirical dt_max (reference ckpt) |
|------|-----------------|--------------------------|-----------------------------------|
| 256  | 1 s             | 0.0094                   | ~0.22 (co-adapted A_real)         |
| 512  | 2 s             | 0.0047                   | ~0.22                             |
| 1024 | 4 s             | 0.0023                   | ~0.22                             |
| 2048 | 8 s             | 0.0012                   | ~0.22                             |

The default `dt_max = 0.1` (from the S4 paper) is actually closer to the empirically
needed range than the theoretical constraint suggests. The theoretical bound is
overly conservative because it ignores A_real co-adaptation.
