# Anti-Collapse Denoising Strategies

Five strategies for denoising whitened BNS gravitational-wave strain at SNR 10.

## The Core Problem

Standard MSE and PSD losses collapse to near-zero predictions. The cause is a
~100× amplitude mismatch: whitened noise amplitude ~10, signal amplitude ~0.1.
The gradient of `MSE(pred, noisy_target)` is dominated by the noise component;
minimizing it means predicting the data mean, which is effectively zero.
PSD loss has the same problem: a flat spectrum (zero signal) is a local minimum.

The five strategies below each break this collapse through a different mechanism.

---

## Strategy A — Dynamic Mixture Loss

**File:** `src/BNSReg/tasks/seq_denoise/s4d_mixture_loss.py`  
**Config:** `configs/seq_denoise/ai4gw@cern/s4d_mixture_snr_10/`  
**SLURM:** `slurm/ai4gw@cern/s4d_mixture_snr_10/seq_denoise.slurm`

**Loss:**
```
L = alpha(t) * MSE(pred, clean) + (1 - alpha(t)) * MSE(|FFT(pred)|, |FFT(clean)|)
```

`alpha` ramps linearly from 0 → 1 over `alpha_epochs=1000` epochs.

**Why it helps:** Pure spectral loss (alpha=0) only cares about frequency
content, so the gradient is nonzero whenever the spectrum shape is wrong — it
does not reward predicting zero. Once the model has learned the rough frequency
shape, alpha ramps toward MSE to recover fine temporal structure.

**Key hyperparameters:** `alpha_start`, `alpha_end`, `alpha_schedule` (`linear`
/ `constant` / `cosine`), `alpha_epochs`.

---

## Strategy B — Joint Denoising + Regression (Two-Head, Staged)

**File:** `src/BNSReg/tasks/seq_denoise/s4d_joint_denoise_reg.py`  
**Loader:** `src/BNSReg/dataloader/joint_denoise_loader.py`  
**Config:** `configs/seq_denoise/ai4gw@cern/s4d_joint_snr_10/`  
**SLURM:** `slurm/ai4gw@cern/s4d_joint_snr_10/seq_denoise.slurm`

**Architecture:**
```
input (B, L, 2)
  → S4ModelSeq2Seq backbone (shared encoder, no decoder output used directly)
      ├─ reconstruction head: Linear(d_model, 2) → MSE vs whitened_signal
      └─ regression head:     mean-pool → Linear(d_model, n_vars*2) → GaussNLL
```

**Loss:**
```
L = MSE(rec, clean) + w(t) * GaussNLL(mean, target, var)

w(t) = 0                                    (epoch < warmup_epochs)
     = min((epoch - warmup_epochs) / reg_ramp_epochs, 1) * max_reg_weight
```

**Why it helps:** The GaussNLL gradient requires the backbone to retain the
frequency/phase structure of the chirp (to predict `chirp_mass` and
`mass_ratio`). Even if reconstruction alone would collapse, the regression head
creates an opposing gradient that keeps the signal features alive. The staged
warmup lets reconstruction stabilize first before the regression signal is
introduced.

**Key hyperparameters:** `n_vars=2`, `warmup_epochs=100`, `max_reg_weight=1.0`,
`reg_ramp_epochs=200`. Target variables: `chirp_mass`, `mass_ratio`.

**Data:** `LitBNSDataJointDenoise` returns `(x_noisy, x_clean, y_target,
z_observed)`. Requires both `injected_data_key` and `sig_only_data_key` in the
HDF5, plus `target_variables` and `observed_variables` in the config.

---

## Strategy C — Contrastive Triplet Loss in PSD Space

**File:** `src/BNSReg/tasks/seq_denoise/s4d_contrastive.py`  
**Loader:** `src/BNSReg/dataloader/triplet_loader.py`  
**Config:** `configs/seq_denoise/ai4gw@cern/s4d_contrastive_snr_10/`  
**SLURM:** `slurm/ai4gw@cern/s4d_contrastive_snr_10/seq_denoise.slurm`

**Triplet:**
```
anchor   = model(whitened_injected)   # denoised output
positive = whitened_signal            # clean signal
negative = whitened_bkg               # background noise (same index)
```

**Loss:**
```
d(a, b)   = mean ||PSD(a) - PSD(b)||²
L_triplet = max(0, d(anchor, pos) - d(anchor, neg) + margin)
L_total   = L_triplet + lambda_rec * MSE(anchor, pos)
```

**Why it helps:** If the model predicts zero, both `d(anchor, pos)` and
`d(anchor, neg)` are large (zero is far from both signal and noise in PSD
space), so `L_triplet` is large. The loss is only small when the output is close
to the signal PSD *and* far from the noise PSD — an explicitly asymmetric
gradient that zero cannot satisfy.

**Key hyperparameters:** `margin=0.1`, `lambda_rec=1.0`.

**Data:** `LitBNSDataTriplet` reads three HDF5 keys per sample:
`whitened_injected`, `whitened_signal`, `whitened_bkg`.

---

## Strategy D — Band-Weighted Spectral Loss

**File:** `src/BNSReg/tasks/seq_denoise/s4d_band_weighted.py`  
**Config:** `configs/seq_denoise/ai4gw@cern/s4d_bandweighted_snr_10/`  
**SLURM:** `slurm/ai4gw@cern/s4d_bandweighted_snr_10/seq_denoise.slurm`

**Loss:**
```
W[f] = sigmoid((f - f_low) / bandwidth) * sigmoid((f_high - f) / bandwidth)
L    = mean_f  W[f] * |FFT(pred)[f] - FFT(clean)[f]|²
```

`W` is a static mask precomputed at init (no learned parameters). At 256 Hz
sample rate the Nyquist is 128 Hz; `f_high` is capped at 100 Hz.

**Why it helps:** BNS chirps sweep ~20–100 Hz in this dataset. The mask
amplifies the gradient inside this band and discounts broadband noise residuals
outside it, so the model is more strongly penalized for missing in-band signal
power than for imperfect noise suppression outside the band.

**Key hyperparameters:** `seq_len=1024`, `sample_rate=256.0`, `f_low=20.0`,
`f_high=100.0`, `bandwidth=10.0`.

---

## Strategy E — Next-Token Prediction

**File:** `src/BNSReg/tasks/seq_denoise/s4d_next_token.py`  
**Config:** `configs/seq_denoise/ai4gw@cern/s4d_next_token_snr_10/`  
**SLURM:** `slurm/ai4gw@cern/s4d_next_token_snr_10/seq_denoise.slurm`

**Task:**
```
input:  whitened_injected[:, :-1, :]   # noisy, drop last sample
target: whitened_signal[:, 1:, :]      # clean, drop first sample
loss:   MSE(model(input), target)
```

**Why it helps:** The model sees past noisy context and must predict the *next
clean sample*. Predicting zero is consistently wrong everywhere the signal is
nonzero — there is no way for the model to satisfy the loss by collapsing. This
is the most structurally robust anti-collapse mechanism.

**Why S4D is ideal:** S4D is a state-space model with a causal recurrence
`h_t = A h_{t-1} + B u_t`, `y_t = C h_t + D u_t`. Its convolutional training
form is naturally causal; no masking (unlike attention) is required. The
hidden state summarizes all past inputs, making it well-suited to this
autoregressive task.

---

## Comparison

| Strategy | Anti-collapse mechanism | Extra data needed | Complexity |
|----------|------------------------|-------------------|------------|
| A · Mixture | Spectral loss is nonzero for wrong spectrum shape | None | Low |
| B · Joint | Regression gradient requires chirp features | `target_variables` | High |
| C · Contrastive | Triplet pushes output away from noise PSD | `whitened_bkg` | Medium |
| D · Band-weighted | In-band gradient amplified; out-of-band discounted | None | Low |
| E · Next-token | Zero prediction is wrong at every signal sample | None | Low |

All runs use `gpu_test` partition (12h limit). Create the log directory before
the first submission (commented `mkdir -p` line in each SLURM file).
