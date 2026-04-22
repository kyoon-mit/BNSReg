# SNR Waveform Loss

Physics-informed auxiliary loss used in `model_s4d_gaussnll_waveform.py`.

## Motivation

The standard GaussianNLL loss treats the problem as pure regression: it measures how well the model's predicted distribution covers the true parameter value, but ignores whether the predicted parameters are **consistent with the observed strain waveform**. The SNR loss adds that consistency check: the predicted chirp mass Mc implies a specific gravitational-wave phase evolution; we measure how well that phase template matches the data.

## PN Phase in the Fourier Domain

At leading post-Newtonian (PN) order, the instantaneous GW frequency evolves as

```
f(τ) = 134 Hz × (1.21 / Mc)^(5/8) × τ^(-3/8)
```

where τ is the time remaining until coalescence and Mc is in solar masses. Integrating the phase `φ = 2π ∫ f dτ` gives the stationary-phase approximation (SPA) Fourier-domain waveform:

```
h̃(f) ∝ f^(-7/6) × exp(i Ψ(f, Mc))

Ψ(f, Mc) = (3/128) × (π Mc_sec f)^(-5/3)
```

where `Mc_sec = Mc × G M_sun / c^3 = Mc × 4.9255×10^{-6} s` converts solar masses to the geometric-unit timescale. The full phase also contains a linear term `2π f t_coal` encoding the unknown time of coalescence.

## De-Chirp: Maximizing Over t_coal Without Knowing It

Computing the matched filter inner product `⟨x | h_Mc⟩` for a specific Mc requires knowing `t_coal` (it appears as a linear phase ramp in frequency). Instead we use the de-chirp approach, which evaluates the inner product **for all possible t_coal simultaneously**:

1. **RFFT** the observed strain: `X(f) = RFFT(x)`
2. **Mask** frequencies below `F_MIN` (default 20 Hz) to avoid the PN singularity at `f → 0` and sub-band noise
3. **Remove PN phase**: multiply by `exp(-i Ψ(f, Mc_pred))`
4. **IRFFT** back to the time domain: the result is

   ```
   z(t) = IRFFT[ X(f) × exp(-i Ψ(f, Mc_pred)) ]
   ```

   If `Mc_pred` matches the true Mc, the chirp is exactly cancelled and `z(t)` collapses to a sharp peak at `t = t_coal`. If Mc is wrong, the residual phase scrambles the IRFFT and the peak is suppressed.

5. **Take the peak**: `ρ_unnorm = max_t |z(t)|` — this is the matched filter SNR at the best-fitting t_coal, achieved without ever specifying t_coal.

## Normalization

By the Cauchy-Schwarz inequality, `|⟨x | h⟩|² ≤ ⟨x|x⟩ × ⟨h|h⟩`. Our template has unit amplitude `|H(f)| = 1` for `f ≥ F_MIN`, so:

```
ρ = peak × L / sqrt(S_xx × S_hh)

S_xx = Σ_f |X(f)|²           (strain power in active band)
S_hh = #{f : f ≥ F_MIN}      (template power; |H| = 1 by construction)
```

The `× L` factor undoes the `1/L` normalization of `torch.fft.irfft`. This gives `ρ ∈ [0, 1]`.

## Loss Formula

```
L = L_GaussNLL + λ × (1 − ρ)
```

- `L_GaussNLL` = standard Gaussian NLL over normalized parameters
- `λ = LAMBDA_SNR = 0.1` (class constant; tune by editing the file)
- `1 − ρ` is minimized when the predicted Mc produces a waveform that best matches the observed strain

## Gradient Path

```
mean[:, 0]  (normalized Mc output)
    → Mc  (unnormalized, M_sun)
    → Mc_sec  (geometric units)
    → Ψ(f, Mc_sec)  (PN phase, differentiable via pow(-5/3))
    → exp(-i Ψ)  (complex rotation, differentiable)
    → dechirped spectrum
    → IRFFT  (linear, differentiable)
    → peak  (amax, differentiable almost everywhere)
    → ρ → snr_loss
```

All operations are differentiable. `torch.fft.irfft` is fully supported by autograd. `amax` passes the gradient through the argmax element (straight-through, standard PyTorch behavior).

## Assumptions and Caveats

- **`chirp_mass` must be `target_variables[0]`** — hard-coded as index 0.
- **`normalize_variables=True` required** — `on_fit_start` reads `var_scales['chirp_mass']` from the dataset.
- **Coherent IFO sum** — both detectors are summed before the FFT, implicitly assuming the same sky-averaged template applies to both (no antenna pattern correction). This is an approximation that simplifies the computation.
- **Amplitude weighting omitted** — the SPA amplitude `f^(-7/6)` is not applied. The template has flat amplitude in the active band. This is not optimal matched filtering but is sufficient to make the loss sensitive to Mc.
- **Only meaningful for chirp-mass estimation** — using this model for mass ratio, distance, etc. is fine (the waveform loss only operates on index 0), but those outputs are still trained with GaussNLL only.
