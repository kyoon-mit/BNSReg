import torch

from numpy.typing import ArrayLike
from lightning.pytorch import LightningModule

def log_GaussianNLLLoss(
        task: LightningModule,   # the module in which this callback is called
        stage: str,              # 'train' or 'val'
        loss: float,             # computed loss
        indiv_mse: ArrayLike,    # individual MSELoss of the outputs
        variance: ArrayLike,     # outputted variance
) -> None:
    if stage not in ('train', 'val'):
        raise ValueError(f'Invalid value for {stage=}. Valid values are \'train\' or \'val\'.')
    task.log(f'{stage}/gaussnll', loss, on_step=False, on_epoch=True, prog_bar=True)
    for i in range(len(indiv_mse)):
        task.log(f'{stage}/mse/out_{i}', indiv_mse[i], on_step=False, on_epoch=True)
        task.log(f'{stage}/sigma_{i}', torch.sqrt(variance[i].mean(dim=0)), on_step=False, on_epoch=True)
    return