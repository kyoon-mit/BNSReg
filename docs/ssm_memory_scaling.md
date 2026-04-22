# Why S4D Runs Out of GPU Memory — and How LinOSS Fixes It

This document explains, from first principles, why S4D consumes so much GPU memory when
sequences get long, and why LinOSS does not.

---

## 1. The problem: everything in machine learning lives as a 2-D (or 3-D) tensor

When training a neural network, the GPU must hold three kinds of things in memory
simultaneously:

1. **Parameters** — the weights W of each layer.
2. **Activations** — the intermediate values produced during the forward pass, which must
   be kept alive until the backward pass can compute gradients through them.
3. **Gradients & optimizer state** — roughly 2–3× the size of the parameters (for Adam).

For most modern networks parameters and optimizer state are small compared to activations.
The activation memory is typically what causes out-of-memory (OOM) errors.

For a sequence model processing a batch of shape `(B, L, H)` — batch size B,
sequence length L, feature dimension H — **every operation that touches the full sequence
must store a tensor proportional to B × L × H**.  For B=16, L=65536 (64 s × 1024 Hz),
H=512:

```
B × L × H × 4 bytes (float32) = 16 × 65536 × 512 × 4 = 2.1 GB
```

This is unavoidable for any model that processes the full sequence.

---

## 2. What S4D does — and where the memory explodes

S4D computes the sequence-to-sequence map via **convolution in the Fourier domain**:

```
y(t) = (K * u)(t)    where  K(t) is the SSM impulse-response kernel
```

**Step 1 — Build the convolution kernel K.**
The kernel is derived from the SSM parameters A (diagonal, `d_state × d_model` entries),
C, and the learnable time-step Δ.  S4DKernel materialises it via a Vandermonde product:

```python
# From s4d.py, S4DKernel.forward:
dtA = A * dt.unsqueeze(-1)          # (H, N//2)  — H = d_model, N//2 = d_state // 2
K   = dtA.unsqueeze(-1) * arange(L) # (H, N//2, L)  <-- the memory culprit
C   = C * (exp(dtA) - 1) / A        # (H, N//2)
K   = 2 * einsum('hn,hnl->hl', C, exp(K)).real  # (H, L)
```

The intermediate `(H, N//2, L)` is **complex64** (8 bytes per element).

| Config | Shape | Memory |
|--------|-------|--------|
| H=32, N=32, L=1024 | (32, 16, 1024) | 4 MB |
| H=256, N=32, L=16384 | (256, 16, 16384) | 512 MB |
| H=512, N=32, L=65536 | **(512, 16, 65536)** | **3.2 GB** |

This 3.2 GB intermediate is allocated **once per layer per forward pass**, regardless
of batch size.  With n_layers=8, the kernel computation alone needs 25 GB.

**Step 2 — FFT convolution.**
After building K, S4D performs:

```python
k_f = rfft(K, n=2L)   # (H, L+1) complex
u_f = rfft(u, n=2L)   # (B, H, L+1) complex
y   = irfft(u_f * k_f)[..., :L]  # (B, H, L)
```

`u_f` is `(B, H, L+1)` complex64 = `16 × 512 × 65537 × 8 = 4.1 GB`.
The pointwise product `u_f * k_f` is another 4.1 GB.

**All together, S4D stores per layer (B=16, H=512, L=65536):**

| Tensor | Shape | Memory |
|--------|-------|--------|
| Input u | (16, 512, 65536) float32 | 2.0 GB |
| Vandermonde intermediate | (512, 16, 65536) complex64 | 3.2 GB |
| u_f (FFT of u) | (16, 512, 65537) complex64 | 4.1 GB |
| k_f * u_f | (16, 512, 65537) complex64 | 4.1 GB |
| Output y | (16, 512, 65536) float32 | 2.0 GB |
| GLU intermediate | (16, 1024, 65536) float32 | 4.1 GB |
| **Total per layer** | | **~20 GB** |

With n_layers=8 stored for backprop (without gradient checkpointing): **~160 GB**.

Neither **mixed precision** nor **gradient checkpointing** escapes the Vandermonde
intermediate.  The tensor `(H, N//2, L)` is allocated fresh inside
`S4DKernel.forward(L)` every time and has no checkpoint-able boundary.

---

## 3. What LinOSS does differently

LinOSS is based on a **forced harmonic oscillator** formulation.  The continuous-time
system is:

```
d²y/dt² + A·y = B·u(t)
y_output(t) = C·y(t) + D·u(t)
```

where **A is a diagonal matrix of oscillation frequencies** (shape P × P, with P ≪ H).

### 3.1 The 2×2 block discretization

Introducing the state z = [y, ẏ] (position and velocity), the discretized LinOSS-IM
step is:

```
schur = 1 / (1 + Δ²·A)
M = [[1 - Δ²·A·schur,   -Δ·A·schur],
     [Δ·schur,           schur      ]]     # 2×2 block, entries each (P,)

z_{t+1} = M @ z_t + driving_force(u_t)
output_t = C @ z_{t+1}[velocity_half].real
```

This is a **sequential recurrence** over L steps.  At each step, the computation is:

```python
z1_new = M11 * z1 + M12 * z2 + F1[t]   # (B, P) complex
z2_new = M21 * z1 + M22 * z2 + F2[t]   # (B, P) complex
```

All tensors are proportional to `B × P`, **not B × H or H × N × L**.

### 3.2 Memory inventory (B=16, H=512, P=32, L=65536)

| Tensor | Shape | Memory |
|--------|-------|--------|
| Input u | (16, 65536, 512) float32 | 2.0 GB |
| Bu (input projected to P-dim) | (16, 65536, 32) complex64 | 256 MB |
| Scan states (all L steps × 2) | L × (16, 32) complex64 × 2 | 256 MB |
| SSM output ys | (16, 65536, 32) complex64 | 256 MB |
| Output (ys projected back to H) | (16, 65536, 512) float32 | 2.0 GB |
| **Total per layer** | | **~5 GB** |

With n_layers=8 and **fp16 mixed precision** (halves float32 tensors):

| Strategy | Activation memory | Feasible on 16 GB? |
|----------|-------------------|-------------------|
| fp32, no checkpointing | ~40 GB | No |
| fp16, no checkpointing | ~20 GB | No |
| fp16 + gradient checkpointing | ~6–8 GB | Yes ✓ |

**Gradient checkpointing** (via `torch.utils.checkpoint`) stores activations only at
block boundaries (~√n_layers blocks).  During backward, it replays the forward pass of
each un-checkpointed block from its nearest checkpoint.  This reduces activation memory
from O(n_layers) to O(√n_layers) at the cost of ~1.4× longer backward pass.

---

## 4. Why the loop is fast despite being sequential

The Python for-loop in `LinOSSLayer._sequential_scan` runs L iterations.  This sounds
slow, but:

1. **Each step operates on (B, P) tensors** — tiny element-wise multiplications, not
   matrix products.  The GPU is occupied at the batch level, not the sequence level.

2. **`torch.compile` eliminates Python overhead.**  Without compilation, each of the
   65536 iterations would incur Python-to-CUDA dispatch overhead (~10 µs each = 0.65 s).
   `torch.compile` traces the loop once, generates a single fused CUDA graph, and runs
   it in one dispatch per forward pass.

3. **S4D's FFT is also O(L log L) sequential** on the GPU; it just happens to be one big
   kernel call instead of L small ones.  LinOSS compiled has comparable wall-clock time
   for moderate L.

---

## 5. Summary

| Property | S4D | LinOSS |
|----------|-----|--------|
| Kernel intermediate | O(H × N × L) = 3.2 GB | None |
| Scan/conv memory | O(B × H × L) via FFT | O(B × L × P), P ≪ H |
| Bottleneck at L=65536, H=512 | ~20 GB / layer | ~5 GB / layer |
| Gradient checkpointing fixes OOM? | Partially (not the Vandermonde) | Yes |
| fp16 + checkpointing on 16 GB? | No | Yes |
| Discretization stability | Guaranteed (S4D eigenvalues) | Guaranteed (IM positive-definite) |
| Physical interpretation | Diagonal state spaces | Forced harmonic oscillators |

For the BNS regression task, with GW signals that are inherently **oscillatory chirps**,
LinOSS's harmonic oscillator prior is also a natural physics-informed inductive bias.

---

## 6. AFRAME and LinOSS

> **Important:** The AFRAME repository at `aframe_linoss/` is named after LinOSS for
> historical reasons, but `libs/architectures/networks/linoss.py` contains
> **JAX/Equinox ResNet1D blocks** — not a LinOSS SSM. There is no PyTorch LinOSS
> implementation in AFRAME.
>
> The PyTorch LinOSS SSM (`src/BNSReg/models/linoss.py`) lives exclusively in BNSReg.
> AFRAME users who need memory-efficient long-sequence training should use
> **gradient checkpointing + fp16 with S4D**, or port BNSReg's LinOSS to AFRAME's
> channels-first convention.

The memory analysis in this document (sections 1–5) applies to the PyTorch S4D and
LinOSS implementations in BNSReg. AFRAME's S4D (`libs/architectures/networks/s4d.py`)
has the identical kernel computation and the same memory profile — the Vandermonde
intermediate `(H, N//2, L)` is the bottleneck in both.
