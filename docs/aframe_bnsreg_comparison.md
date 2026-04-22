# BNSReg vs AFRAME: Architecture, Pipeline, and Sensitive Volume

**Last updated:** 2026-04-22

---

## Contents

1. [S4D Implementation Comparison](#1-s4d-implementation-comparison)
2. [LinOSS Implementation Comparison](#2-linoss-implementation-comparison)
3. [Training Pipeline: BNSReg](#3-training-pipeline-bnsreg)
4. [Training Pipeline: AFRAME](#4-training-pipeline-aframe)
5. [Apples-to-Apples Pipeline Comparison](#5-apples-to-apples-pipeline-comparison)
6. [Using σ(Mc) as Detection Statistic in AFRAME](#6-using-σmc-as-detection-statistic-in-aframe)
7. [Comparison with AFRAME Benchmarks](#7-comparison-with-aframe-benchmarks)

---

## 1. S4D Implementation Comparison

Both repos derive from the same S4D paper (Gu et al. 2022). The core Vandermonde kernel
and FFT convolution are identical. Differences are minor:

| Aspect | BNSReg (`src/BNSReg/models/s4d.py`) | AFRAME (`libs/architectures/networks/s4d.py`) |
|--------|--------------------------------------|-----------------------------------------------|
| Input convention | channels-last `(B, L, d_input)` | channels-first `(B, d_input, L)` |
| `input_norm` flag | yes — optional `InstanceNorm1d` on strain channels | no |
| `prenorm` option | yes | no (postnorm only) |
| `torch.compile` | applied to `S4Model` in task class | not applied by default |
| Type annotation style | `Optional[float]` | `float \| None` |

The S4D kernel itself is bit-for-bit identical:

```python
# Both repos — S4DKernel.forward(u, L):
dtA = A * dt.unsqueeze(-1)           # (H, N//2) complex
K   = dtA.unsqueeze(-1) * arange(L) # (H, N//2, L)  ← memory bottleneck
C   = C * (exp(dtA) - 1) / A
K   = 2 * einsum('hn,hnl->hl', C, exp(K)).real
k_f = rfft(K, n=2L)
y   = irfft(rfft(u) * k_f)[..., :L] + u * D
```

**Channel convention note.** BNSReg's task model transposes before calling the model:

```python
# model_s4d_gaussnll.py — training_step:
X_sequence = X_sequence.transpose(2, 1)  # (B, 2, L) → (B, L, 2)
```

AFRAME's model accepts `(B, 2, L)` directly and transposes internally inside `S4Model.forward`.
The net computation is the same.

---

## 2. LinOSS Implementation Comparison

> **Critical note:** The AFRAME repository is named `aframe_linoss` for historical reasons,
> but `libs/architectures/networks/linoss.py` contains **JAX/Equinox ResNet1D blocks**,
> not a LinOSS SSM. There is no PyTorch LinOSS implementation in AFRAME.
> The PyTorch LinOSS SSM lives exclusively in BNSReg.

### BNSReg LinOSS (`src/BNSReg/models/linoss.py`)

State-space formulation:

```
d²y/dt² + A·y = B·u(t)    A diagonal (P×P), A > 0 enforced via relu
output(t) = C·ẏ(t) + D·u(t)
```

Discretization (IM — implicit midpoint):

```
schur = 1 / (1 + Δ²·A)
M11 = 1 − Δ²·A·schur    M12 = −Δ·A·schur
M21 = Δ·schur            M22 = schur

z1_{t+1} = M11·z1_t + M12·z2_t + F1_t
z2_{t+1} = M21·z1_t + M22·z2_t + F2_t
```

Key parameters:

| Parameter | Shape | Initialization |
|-----------|-------|----------------|
| `A_raw` | `(P,)` | uniform, relu → positive frequencies |
| `B_raw` | `(P, H, 2)` | `N(0, 1/√H)` complex stored as real |
| `C_raw` | `(H, P, 2)` | `N(0, 1/√P)` complex stored as real |
| `D` | `(H,)` | feedthrough (skip connection) |
| `step_raw` | `(P,)` | sigmoid → learned step size per oscillator |

The sequential scan is compiled by `torch.compile` into a single CUDA graph, making
wall-clock time comparable to S4D's FFT for practical sequence lengths (L ≤ 16384).

### AFRAME "linoss.py" (actually JAX ResNet1D)

The file at `libs/architectures/networks/linoss.py` is a JAX/Equinox CNN with:
- `ResNet1D` blocks (1D convolutions + batch norm + residual)
- No recurrence, no state-space structure
- Used only for classification baselines in the JAX-based training path

This is unrelated to the harmonic oscillator SSM. Do not confuse the two.

---

## 3. Training Pipeline: BNSReg

```
YAML config
    │
    ├─ model:   class_path + init_args  ─────────────────────────────────────┐
    ├─ data:    class_path + init_args  ─────────────────────────────────────┤
    └─ trainer: accelerator, callbacks, logger, epochs                       │
                                                                             │
FitTestCLI (LightningCLI)                                                    │
    │                                                                         │
    ├─ DataModule (LitBNSDataRegression or LitBNSDataCurriculum)             │
    │       │                                                                 │
    │       ├─ __init__: reads HDF5, computes var_scales if normalize=True   │
    │       ├─ train_dataset: BNSDatasetRegression(train_file)               │
    │       ├─ val_dataset:   BNSDatasetRegression(val_file)                 │
    │       └─ test_dataset:  BNSDatasetRegression(test_file)                │
    │               │                                                         │
    │               └─ __getitem__(idx):                                      │
    │                     HDF5[whitened_injected][idx] → X (L, 2)            │
    │                     HDF5[chirp_mass][idx]        → y_target            │
    │                     HDF5[snr][idx]               → z_observed          │
    │                     if normalize: y → [0, 1]                           │
    │                     return (X, y_target, z_observed)                   │
    │                                                                         │
    └─ LitModelS4DGaussianNLLLoss                                            │
            │                                                                 │
            ├─ configure_model:                                               │
            │       S4Model(d_input=2, d_output=2*n_vars, ...)               │
            │       torch.compile(model)                                      │
            │                                                                 │
            ├─ configure_optimizers:                                          │
            │       pg1: non-SSM params at base_lr with weight_decay         │
            │       pg2: SSM params (log_dt, log_A_real, A_imag) at lr       │
            │       WarmupCosineAnnealingWarmRestarts                        │
            │                                                                 │
            ├─ training_step(batch):                                          │
            │       X.T → (B, L, 2)                                          │
            │       outputs = model(X) → (B, 2*n_vars)                       │
            │       mean = outputs[:, :n_vars]                                │
            │       var  = softplus(outputs[:, n_vars:])                      │
            │       nll  = BetaNLLLoss(β)(mean, y_target, var)               │
            │       spread = softplus(Var(y_true) − Var(mean))               │
            │       loss = nll + λ_spread × spread                            │
            │       logs: train/gaussnll, train/mse/out_*, train/loss        │
            │                                                                 │
            ├─ validation_step: same, logs val/gaussnll, val/mse/out_*       │
            │                                                                 │
            └─ test_step:                                                     │
                    returns {y_true, y_pred, y_sigma, z_observed}            │
                    PlotParamEstCallback saves PNG + CSV                      │

After fit: FitTestCLI.after_fit() runs trainer.test(best_ckpt)
```

**Data is pre-whitened** and stored in HDF5. No on-the-fly whitening or injection.

---

## 4. Training Pipeline: AFRAME

```
Luigi DAG (aframe/tasks/)
    │
    ├─ FetchBackground ──────────────────────────────────────────────────┐
    │       downloads raw strain from GWOSC or local cache               │
    │       segments by data quality flags                               │
    │       output: background HDF5 files                               │
    │                                                                    │
    ├─ FetchTrain (requires FetchBackground)                            │
    │       computes PSD from background                                 │
    │       stores training background segments                         │
    │                                                                    │
    ├─ TrainingWaveforms (requires FetchBackground + prior)             │
    │       samples CBC parameters from prior                           │
    │       generates IMRPhenomD waveforms                              │
    │       output: training_waveforms.hdf5                            │
    │                                                                    │
    ├─ ValidationWaveforms (requires FetchBackground)                  │
    │       same as above for fixed validation set                      │
    │       output: val_waveforms.hdf5                                  │
    │                                                                    │
    └─ TrainLocal (requires all above)                                  │
            │                                                            │
            └─ subprocess: python -m train fit --config regression_s4d.yaml
                    │
                    ├─ RegressionTimeDomainDataset (LightningDataModule)
                    │       background_dir: raw strain HDF5
                    │       waveforms_dir:  pre-computed waveforms HDF5
                    │       
                    │       per batch (on the fly):
                    │         1. sample background segment (B, 2, L)
                    │         2. compute whitening filter from PSD
                    │         3. whiten background
                    │         4. sample waveform at random SNR (PowerLaw prior)
                    │         5. whiten waveform template
                    │         6. inject: x_inj = x_bg + scale * x_template
                    │         7. return (x_inj, y_params, empty)
                    │
                    └─ LitS4DGaussianNLL (LightningModule)
                            │
                            ├─ S4Model(d_input=2, channels-first)
                            ├─ BetaNLLLoss(β=0.5)
                            ├─ optimizer: single AdamW group, lr=1e-4
                            ├─ training_step: same NLL loss structure
                            └─ validation_step: val/gaussnll, val/mse

After training: inference pipeline (aframe/tasks/infer/) runs on test segments
producing EventSet HDF5 → fed to plots/sv.py for sensitive volume calculation
```

---

## 5. Apples-to-Apples Pipeline Comparison

| Step | BNSReg | AFRAME |
|------|--------|--------|
| **Data generation** | Pre-processed HDF5 (external) | Luigi DAG fetches raw strain + generates waveforms |
| **Whitening** | Pre-computed, stored in HDF5 | On-the-fly per batch from PSD of background |
| **Injection** | Pre-injected into HDF5 | On-the-fly at random SNR drawn from PowerLaw prior |
| **SNR curriculum** | Stage-weighted mixture of HDF5 files (discrete bins) | Continuous PowerLaw SNR sampler with exponential decay |
| **Input convention** | channels-last `(B, L, 2)` — transpose in task | channels-first `(B, 2, L)` — transpose inside S4Model |
| **Optimizer** | Two param groups: SSM (lr=1e-3) + default (lr=1e-4) | Single group: lr=1e-4, wd=0 |
| **Validation metric** | `val/mse/out_0`, `val/gaussnll` | `val/gaussnll`, `val/mse/out_*` (identical) |
| **Classification metric** | None — regression only | `TimeSlideAUROC` (AUROC at fixed max FPR, timeslide-aware) |
| **Test output** | CSV + PNG plots via `PlotParamEstCallback` | `EventSet` HDF5 → sensitive volume plots |
| **Output head scale** | Variables normalized to [0,1] (via `normalize_variables`) | Raw physical units (chirp mass in M☉) |
| **torch.compile** | Applied to entire `S4Model` | Not applied by default |

**Key insight — BNSReg vs AFRAME data flow:**

- BNSReg trains on a **fixed HDF5 dataset** generated offline. The SNR distribution is
  determined by the dataset generator. Curriculum is achieved by mixing files from
  different SNR bins.
- AFRAME **generates batches on the fly** from raw background + waveform templates, with
  the SNR for each injection drawn fresh each step. This is closer to the online training
  scheme used for production aLIGO search pipelines.

---

## 6. Using σ(Mc) as Detection Statistic in AFRAME

The AFRAME sensitive volume pipeline expects a scalar **detection statistic** per event —
typically network SNR or a matched-filter score. To use the BNSReg model's predicted
uncertainty σ(Mc) (or the full GaussNLL score) as the detection statistic, follow these steps.

### 6.1 Concept

A well-calibrated regression model will assign low σ(Mc) to events where the signal is
clearly resolvable (high SNR, strong signal) and high σ(Mc) to noise or weak events.
The detection statistic is then:

```
stat(event) = −σ(Mc)        (lower σ → more confidently detected → higher stat)
```

or equivalently the negative GaussNLL evaluated on the event:

```
stat(event) = −NLL(Mc_pred, σ_pred | strain)
```

The latter is the log-likelihood ratio under the model, making it the theoretically
optimal detection statistic by the Neyman-Pearson lemma (assuming the model is calibrated).

### 6.2 Inference Step

1. **Train the BNSReg regression model** on the SNR 15-30 dataset as usual.
   Save the best checkpoint (monitored on `val/mse/out_0`).

2. **Run inference on AFRAME test events.** AFRAME's inference pipeline
   (`aframe/tasks/infer/`) processes strain segments and produces a score per event.
   Replace the score computation with a call to your trained model:

   ```python
   # In AFRAME's infer pipeline — replace default scoring with:
   from BNSReg.tasks.parameter_estimation.model_s4d_gaussnll import LitModelS4DGaussianNLLLoss

   model = LitModelS4DGaussianNLLLoss.load_from_checkpoint(ckpt_path)
   model.eval()

   with torch.no_grad():
       outputs = model.model(strain_batch)          # (B, 2*n_vars)
       mean  = outputs[:, :n_vars]
       var   = model.var_activation(outputs[:, n_vars:])
       sigma = var.sqrt()

       # Option A: use −σ(Mc) as detection stat
       detection_stat = -sigma[:, 0]               # (B,)

       # Option B: use −NLL as detection stat (optimal)
       nll = 0.5 * (var.log() + (mean - y_ref)**2 / var)
       detection_stat = -nll.mean(dim=1)           # (B,)
   ```

   For a pure detection task (no reference Mc available at inference time),
   **use Option A** (−σ). For events with known injected parameters (validation set),
   **Option B** is optimal.

3. **Write detection statistics to EventSet HDF5.** AFRAME's ledger system stores
   events as `RecoveredInjectionSet` (foreground) and `EventSet` (background).
   Add a field `sigma_mc` or `detection_stat` to these:

   ```python
   from ledger.injections import RecoveredInjectionSet
   foreground = RecoveredInjectionSet.read(foreground_path)
   foreground['detection_stat'] = detection_stats_for_injections
   foreground.write(output_path)
   ```

### 6.3 Sensitive Volume Computation

Once detection statistics are in the `EventSet` HDF5, AFRAME's existing
`sensitive_volume` function handles everything:

```python
# projects/plots/plots/vizapp/pages/summary/compute.py
from plots.vizapp.pages.summary.compute import sensitive_volume

# detection_statistics: shape (n_events,) — your −σ(Mc) values
# weights: shape (n_events,) — prior weights (distance⁻² × cosm. jacobian)
# thresholds: grid of stat values (FAR scan)

sv, sv_err = sensitive_volume(
    detection_statistics=foreground_stats,
    weights=foreground_weights,
    thresholds=background_thresholds_at_target_far,
)
```

The output is a `(n_sources, n_thresholds)` array of sensitive volumes with
statistical uncertainties, directly comparable to AFRAME's SNR-based SV curves.

### 6.4 Running the Full Pipeline

```bash
# 1. Train BNSReg model
python -m BNSReg.core.main_fit_test fit \
    --config configs/parameter_estimation/checks/s4d_gaussnll_snr_15_30/chirp_mass_59-63s_d64_s32_l4_paramnorm.yaml

# 2. Export test predictions to CSV (PlotParamEstCallback does this automatically)
#    CSV will contain: chirp_mass_true, chirp_mass_pred, sigma_chirp_mass_pred, snr
#    File location: outputs/<project>/<run_id>/plots/param_est_results.csv

# 3. Map CSV predictions to AFRAME EventSet format
python scripts/export_to_aframe_ledger.py \
    --csv outputs/.../param_est_results.csv \
    --foreground /path/to/aframe/foreground.hdf5 \
    --output /path/to/scored_foreground.hdf5

# 4. Run AFRAME sensitive volume calculation
cd /path/to/aframe_linoss
python -m plots.matplotlib.main \
    --background background.hdf5 \
    --foreground scored_foreground.hdf5 \
    --output-dir sv_plots/
```

---

## 7. Comparison with AFRAME Benchmarks

AFRAME's default detection statistic is **network SNR** (matched-filter optimal for
Gaussian noise). Two relevant comparisons:

### 7.1 SNR vs. σ(Mc) as Detection Statistic

| Statistic | Assumptions | Advantages | Disadvantages |
|-----------|-------------|------------|---------------|
| Network SNR | Gaussian noise, known template | Optimal under Neyman-Pearson in Gaussian noise | Requires template bank; computationally expensive |
| −σ(Mc) | Model is calibrated; signal has extractable Mc | Template-free; model learns noise structure directly | Requires training data; suboptimal if model mismatch |
| −NLL(Mc) | Same as above + known prior | Log-likelihood ratio — theoretically optimal for model | Requires reference Mc (only valid on known injections) |

### 7.2 TimeSlideAUROC Benchmark

AFRAME's validation metric is `TimeSlideAUROC` — AUROC evaluated on timeslide background
up to a max FPR (e.g., 1%). To compare your σ(Mc)-based statistic:

```python
from train.metrics import TimeSlideAUROC

metric = TimeSlideAUROC(max_fpr=1e-2, stride=0.5, pool_length=128)
metric.update(shift=0, background=bg_stats, foreground=fg_stats)
auroc = metric.compute()
print(f"AUROC @ FAR<1%: {auroc:.4f}")
```

A good regression-based statistic should achieve AUROC ≥ 0.85 at SNR 15-30.
AFRAME's matched-filter baseline is typically 0.92-0.97 in the same regime.

### 7.3 SV Curve Comparison Protocol

To produce a fair apples-to-apples SV comparison:

1. Use the **same background segments** for both pipelines (identical FAR denominator).
2. Use the **same injection set** (identical foreground numerator — same Mc, q, distance).
3. Scan the **same FAR range** (e.g., 1/year to 1/hour).
4. Plot SV(FAR) curves on the same axes.

The sensitive volume at FAR = 1/month is the standard figure of merit used in
LIGO O4 searches and AFRAME papers. At SNR 15-30, expect the regression-based
statistic to be within 10-20% of the matched-filter SV if the model is well-trained.
