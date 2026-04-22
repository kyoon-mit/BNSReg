"""β-NLL loss for heteroscedastic regression.

Seitzer et al. (2022), "On the Pitfalls of Heteroscedastic Uncertainty
Estimation with Probabilistic Neural Networks."
https://arxiv.org/abs/2203.09168

Standard GaussianNLL has a degenerate local minimum where the model can
inflate predicted variance to suppress the mean gradient:

    ∂NLL/∂mean = (mean − y) / var  →  0  as  var → ∞

β-NLL weights the NLL by stop_gradient(var)^β:

    L_β = sg(var)^β · L_NLL
    ∂L_β/∂mean = sg(var)^(β−1) · (mean − y)

With β=0 → standard NLL (degenerate); β=0.5 → gradient ∝ 1/std (recommended);
β=1 → pure MSE gradient, fully decoupled from variance. The paper recommends
β=0.5 as a robust default.
"""

import torch
import torch.nn as nn


class BetaNLLLoss(nn.Module):
    """Heteroscedastic β-NLL loss (Seitzer et al. 2022).

    Args:
        beta:      Weighting exponent in [0, 1].  β=0 is standard NLL;
                   β=0.5 (default) is the paper's recommendation; β=1
                   gives an MSE-like mean gradient independent of variance.
        reduction: 'mean' (default) or 'sum'.
    """

    def __init__(self, beta: float = 0.5, reduction: str = 'mean'):
        super().__init__()
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f'beta must be in [0, 1], got {beta}')
        if reduction not in ('mean', 'sum'):
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction}")
        self.beta = beta
        self.reduction = reduction

    def forward(self, mean: torch.Tensor, target: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mean:   predicted mean,     shape (B, n_vars)
            target: ground-truth target, shape (B, n_vars)
            var:    predicted variance,  shape (B, n_vars), must be > 0

        Returns:
            scalar loss
        """
        # Element-wise NLL: 0.5 * [log(var) + (mean − y)² / var]
        nll = 0.5 * (torch.log(var) + (mean - target) ** 2 / var)  # (B, n_vars)

        if self.beta > 0.0:
            # Weight by sg(var)^β so the mean gradient scales as sg(var)^(β−1)
            # rather than 1/var, preventing variance inflation from suppressing it.
            nll = nll * var.detach().pow(self.beta)

        return nll.mean() if self.reduction == 'mean' else nll.sum()
