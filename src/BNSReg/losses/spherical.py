import torch
import torch.nn as nn

# H1-L1 baseline unit vector in the Earth-fixed frame used by ml4gw.
# Derived from ml4gw.gw.get_ifo_geometry('H1', 'L1'): (vertices[1] - vertices[0]).normalize()
H1L1_BASELINE = torch.tensor([0.6953014731407166, -0.5535351634025574, -0.4584263563156128])


def sky_angles_to_vec(dec: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """(dec, phi) in radians → unit 3-vector. Output shape: (*batch, 3)."""
    x = torch.cos(dec) * torch.cos(phi)
    y = torch.cos(dec) * torch.sin(phi)
    z = torch.sin(dec)
    return torch.stack([x, y, z], dim=-1)


def ring_distance(v_pred: torch.Tensor, v_true: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
    """Mean |cos(θ_pred) - cos(θ_true)| where θ is the angle to the H1-L1 baseline.
    cos(θ) = x̂_sky · r̂_baseline encodes the time delay; small ring_distance means
    the model predicts the correct ring even if it's on the wrong point of the ring.
    v_pred, v_true: (B, 3) unit vectors. baseline: (3,) unit vector.
    """
    b = baseline.to(v_pred.device)
    cos_pred = (v_pred * b).sum(dim=-1)   # (B,)
    cos_true = (v_true * b).sum(dim=-1)   # (B,)
    return (cos_pred - cos_true).abs().mean()


class CosineSkyLoss(nn.Module):
    """Great-circle distance loss: 1 − (v̂_pred · v̂_true).

    pred:          (B, 3) raw logits — L2-normalised internally.
    target_angles: (B, 2) = [dec, phi] in radians (NOT normalised).
    Returns scalar in [0, 2]; random baseline ≈ 1.
    """

    def forward(self, pred: torch.Tensor, target_angles: torch.Tensor) -> torch.Tensor:
        v_pred = pred / (pred.norm(dim=-1, keepdim=True) + 1e-8)
        v_true = sky_angles_to_vec(target_angles[:, 0], target_angles[:, 1])
        cos_sim = (v_pred * v_true).sum(dim=-1)  # (B,)
        return (1.0 - cos_sim).mean()
