import torch

from BNSReg.tasks.base_task import LitDenoiserBaseTask
from BNSReg.models.s4d import S4Model
from BNSReg.core.config import S4DDenoisedRegressionConfig
from BNSReg.callbacks.log_metric import log_GaussianNLLLoss


class LitModelDenoisedS4DGaussianNLLLoss(LitDenoiserBaseTask):
    """GaussianNLL regression with an optional frozen on-the-fly denoiser.

    The denoiser (S4ModelSeq2Seq) is loaded from `model_cfg.denoiser_ckpt`
    and `model_cfg.denoiser_cfg` and applied under torch.no_grad() before
    the regression model sees each batch.  When both paths are None the
    model behaves identically to LitModelS4DGaussianNLLLoss.

    Input batch: (X_sequence, y_target, z_observed)
        X_sequence: (B, n_ifos, L)
        y_target:   (B, n_vars)
        z_observed: (B, n_observed)
    """

    def __init__(self, model_cfg: S4DDenoisedRegressionConfig):
        super().__init__()
        if model_cfg.d_output % 2 != 0:
            raise ValueError(
                f'{model_cfg.d_output=} must be even (mean + var for each variable).'
            )
        self.save_hyperparameters()
        self.cfg       = model_cfg
        self.n_vars    = model_cfg.d_output // 2
        self.criterion = torch.nn.GaussianNLLLoss(reduction='mean')
        self.softplus  = torch.nn.Softplus()
        self.model     = None
        self.configure_model()
        self._setup_denoiser(model_cfg.denoiser_ckpt, model_cfg.denoiser_cfg)

    def configure_model(self):
        if self.model is not None:
            return
        self.model = S4Model(**self.cfg.model_kwargs())
        self.model = torch.compile(self.model)

    def forward(self, x):
        return self.model(x)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_outputs(self, outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split model output into (mean, var)."""
        mean = outputs[:, :self.n_vars]
        var  = self.softplus(outputs[:, self.n_vars:])
        return mean, var

    @staticmethod
    def _per_var_mse(mean: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        """Per-variable MSE averaged over the batch. Returns (n_vars,)."""
        return (mean - y_target).pow(2).mean(dim=0)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def compute_loss(self, batch):
        X_sequence, y_target, z_observed = batch
        outputs     = self(self._prepare_input(X_sequence))
        mean, var   = self._split_outputs(outputs)
        loss        = self.criterion(mean, y_target, var)
        y_indiv_mse = self._per_var_mse(mean, y_target)
        return loss, y_indiv_mse, var, mean, y_target

    def training_step(self, batch, batch_idx):
        loss, y_indiv_mse, var, mean, y_target = self.compute_loss(batch)
        log_GaussianNLLLoss(self, 'train', loss, y_indiv_mse, var)
        return {'loss': loss, 'mean': mean.detach(), 'y_target': y_target.detach()}

    def validation_step(self, batch, batch_idx):
        loss, y_indiv_mse, var, mean, y_target = self.compute_loss(batch)
        log_GaussianNLLLoss(self, 'val', loss, y_indiv_mse, var)
        return {'loss': loss, 'mean': mean.detach(), 'y_target': y_target.detach()}

    def test_step(self, batch, batch_idx):
        X_sequence, y_target, _ = batch
        outputs   = self(self._prepare_input(X_sequence))
        mean, var = self._split_outputs(outputs)
        return {
            'y_true':  y_target.detach().cpu(),
            'y_pred':  mean.detach().cpu(),
            'y_sigma': var.sqrt().detach().cpu(),
        }
