import torch
import torch.nn.functional as F
from torch import optim

from BNSReg.tasks.base_task import LitBaseTask
from BNSReg.models.s4d_seq2seq import S4ModelSeq2Seq
from BNSReg.core.config import S4DModelConfig


class LitModelS4DNextToken(LitBaseTask):
    """S4D denoiser trained via next-token prediction on clean signal.

    Standard reconstruction losses (MSE, PSD) collapse toward zero because
    the noise amplitude (~10) dominates the signal (~10⁻¹).  By shifting
    the target one step into the future the model is forced to predict a
    non-zero clean sample at every position, making zero-collapse an
    inherently high-loss solution.

    Task:
        Input:  noisy_injected[:, :-1, :]   (B, L-1, n_ifos)
        Target: clean_signal  [:, 1:,  :]   (B, L-1, n_ifos)

    The S4D SSM is naturally causal (y_t depends only on u_0 … u_t), so
    no masking is needed and the convolutional training path is unmodified.

    Input/output shapes follow LitModelS4DAE exactly (batch from
    BNSDatasetSeqEncoder returns (injected, signal) pairs of shape
    (B, n_ifos, L)).
    """

    def __init__(self, cfg: S4DModelConfig):
        super().__init__()
        self.cfg = cfg
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.model = None
        self.configure_model()

    def configure_model(self):
        if self.model is not None:
            return
        self.model = S4ModelSeq2Seq(**self.cfg.model_kwargs())
        self.model = torch.compile(self.model)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage: str):
        input, target = batch            # (B, n_ifos, L)
        x = input.transpose(1, 2)       # (B, L, d_input)  — noisy
        t = target.transpose(1, 2)      # (B, L, d_output) — clean

        # Next-token: model sees noisy context [0:L-1], predicts clean [1:L]
        x_ctx = x[:, :-1, :]            # (B, L-1, d_input)
        t_nxt = t[:, 1:,  :]            # (B, L-1, d_output)
        out   = self(x_ctx)             # (B, L-1, d_output)

        loss = self.criterion(out, t_nxt)
        self.log(f'{stage}/loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        n_ifos = out.shape[-1]
        for i in range(n_ifos):
            mse_i = self.criterion(out[..., i], t_nxt[..., i])
            self.log(f'{stage}/mse/ifo_{i}', mse_i, on_step=False, on_epoch=True)

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
