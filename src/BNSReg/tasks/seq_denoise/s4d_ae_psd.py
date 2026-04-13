import torch
import torch.nn.functional as F
from torch import optim

from BNSReg.tasks.base_task import LitBaseTask
from BNSReg.models.s4d_seq2seq import S4ModelSeq2Seq
from BNSReg.core.config import S4DModelConfig


class LitModelS4DAE_PSD(LitBaseTask):
    """S4D autoencoder trained with a power-spectral-density (PSD) MSE loss.

    The loss is computed in the frequency domain: for each IFO channel the
    one-sided power spectrum of the predicted and target waveforms is estimated
    via rfft, and the mean squared error between the two PSDs is minimised.
    This encourages the model to reproduce the correct spectral shape rather
    than exact time-domain point values.

    Input/output shapes follow LitModelS4DAE exactly:
        batch: (B, n_ifos, L), (B, n_ifos, L)
        model: (B, L, d_input=n_ifos) -> (B, L, d_output=n_ifos)
    """

    def __init__(self, cfg: S4DModelConfig):
        super().__init__()
        self.cfg = cfg
        self.model = None
        self.configure_model()

    def configure_model(self):
        if self.model is not None:
            return
        self.model = S4ModelSeq2Seq(**self.cfg.model_kwargs())
        self.model = torch.compile(self.model)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def _psd_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MSE between one-sided power spectra.

        Args:
            pred:   (B, L, n_ifos)
            target: (B, L, n_ifos)
        Returns:
            scalar loss
        """
        # rfft along the time axis
        pred_psd   = torch.fft.rfft(pred,   dim=1).abs().pow(2)   # (B, L//2+1, n_ifos)
        target_psd = torch.fft.rfft(target, dim=1).abs().pow(2)
        return F.mse_loss(pred_psd, target_psd)

    def _step(self, batch, stage: str):
        input, target = batch        # (B, n_ifos, L)
        x = input.transpose(1, 2)   # (B, L, d_input)
        t = target.transpose(1, 2)  # (B, L, d_output)
        out = self(x)               # (B, L, d_output)

        loss = self._psd_mse(out, t)
        self.log(f'{stage}/psd_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        n_ifos = out.shape[-1]
        for i in range(n_ifos):
            psd_i = self._psd_mse(out[..., i:i+1], t[..., i:i+1])
            self.log(f'{stage}/psd_loss/ifo_{i}', psd_i, on_step=False, on_epoch=True)

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
