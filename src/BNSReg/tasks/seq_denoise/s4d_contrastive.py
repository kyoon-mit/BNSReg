import torch
import torch.nn.functional as F
from torch import optim

from BNSReg.losses.denoising import TripletSpectralLoss
from BNSReg.tasks.base_task import LitBaseTask
from BNSReg.models.s4d import S4ModelSeq2Seq
from BNSReg.core.config import S4DModelConfig


class LitModelS4DContrastive(LitBaseTask):
    """S4D denoiser trained with a contrastive triplet loss in PSD space.

    Standard reconstruction losses collapse to zero because the noise
    amplitude dominates by ~100×.  The triplet loss adds an asymmetric
    gradient: the denoised output (anchor) is pushed toward the clean
    signal (positive) AND explicitly pushed away from background-only
    noise (negative).  If the model predicts zero, both distances are
    large — unlike MSE where predicting zero is rewarded on the noise.

    Loss:
        d(a, b) = mean ||PSD(a) - PSD(b)||²
        L_triplet = max(0, d(anchor, pos) - d(anchor, neg) + margin)
        L_total   = L_triplet + lambda_rec * MSE(anchor, pos)

    Data: requires LitBNSDataTriplet (returns x_injected, x_signal, x_bkg).

    Args:
        cfg:        S4D model configuration.
        margin:     Triplet margin. Start small (0.1) and tune.
        lambda_rec: Weight on reconstruction MSE for training stability.
    """

    def __init__(
        self,
        cfg:        S4DModelConfig,
        margin:     float = 0.1,
        lambda_rec: float = 1.0,
    ):
        super().__init__()
        self.cfg       = cfg
        self.criterion = TripletSpectralLoss(margin=margin, lambda_rec=lambda_rec)
        self.model     = None
        self.configure_model()

    def configure_model(self):
        if self.model is not None:
            return
        self.model = S4ModelSeq2Seq(**self.cfg.model_kwargs())
        self.model = torch.compile(self.model)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage: str):
        x_inj, x_sig, x_bkg = batch         # each (B, n_ifos, L)

        x   = x_inj.transpose(1, 2)         # (B, L, d_input)
        pos = x_sig.transpose(1, 2)         # (B, L, n_ifos)
        neg = x_bkg.transpose(1, 2)         # (B, L, n_ifos)

        anchor = self(x)                    # (B, L, d_output = n_ifos)

        loss = self.criterion(anchor, pos, neg)
        self.log(f'{stage}/loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        # Diagnostic: log component losses
        a_psd = TripletSpectralLoss._psd(anchor)
        p_psd = TripletSpectralLoss._psd(pos)
        n_psd = TripletSpectralLoss._psd(neg)
        d_pos = (a_psd - p_psd).pow(2).mean()
        d_neg = (a_psd - n_psd).pow(2).mean()
        self.log(f'{stage}/dist_pos', d_pos, on_step=False, on_epoch=True)
        self.log(f'{stage}/dist_neg', d_neg, on_step=False, on_epoch=True)
        self.log(f'{stage}/mse',      F.mse_loss(anchor, pos),
                 on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._step(batch, 'test')

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }
