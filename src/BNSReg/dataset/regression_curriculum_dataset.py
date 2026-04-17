import torch
from torch.utils.data import Dataset

from BNSReg.core.config import BNSDataModuleCurriculumConfig
from BNSReg.dataset.regression_dataset import BNSDatasetRegression


class BNSDatasetMixture(Dataset):
    """Draws samples from multiple BNSDatasetRegression instances with mixture weights.

    On each __getitem__ call, a source dataset is chosen by multinomial sampling
    over the current weights, then an index is drawn uniformly from that dataset.
    This means the effective epoch size equals __len__ = sum of all dataset sizes,
    but the distribution over samples follows the mixture weights.

    Update weights between epochs via set_weights(); set a weight to 0.0 for
    hard exclusion of a file (Option A / hard-switching behaviour).
    """

    def __init__(
        self,
        datasets: list[BNSDatasetRegression],
        weights: list[float],
    ) -> None:
        if len(datasets) != len(weights):
            raise ValueError("datasets and weights must have the same length.")
        self.datasets = datasets
        self._weights = torch.tensor(weights, dtype=torch.float32)

    def set_weights(self, weights: list[float]) -> None:
        self._weights = torch.tensor(weights, dtype=torch.float32)

    @property
    def weights(self) -> torch.Tensor:
        return self._weights / self._weights.sum()

    def __len__(self) -> int:
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, idx: int):
        ds_idx = int(torch.multinomial(self.weights, num_samples=1).item())
        sample_idx = idx % len(self.datasets[ds_idx])
        return self.datasets[ds_idx][sample_idx]
