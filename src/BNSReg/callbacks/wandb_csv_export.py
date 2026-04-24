from pathlib import Path

import pandas as pd
import wandb
from lightning.pytorch.callbacks import Callback


class WandBCSVExportCallback(Callback):
    """Export the full W&B run history to CSV at the end of the test epoch.

    Saves to {logger.save_dir}/{project}/{run_id}/metrics.csv — the same
    base directory that holds the checkpoints/ and plots/ subdirectories.
    """

    def on_test_end(self, trainer, pl_module):
        logger = trainer.logger
        if logger is None or not hasattr(logger, 'experiment'):
            return

        exp = logger.experiment
        entity  = getattr(exp, 'entity',  None)
        project = getattr(exp, 'project', None)
        run_id  = getattr(exp, 'id',      None)

        if not all([entity, project, run_id]):
            return

        # Resolve output path: same base as plots/ and checkpoints/
        save_dir = getattr(logger, 'save_dir', None)
        if save_dir is None:
            return
        out_dir = Path(save_dir) / project / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'metrics.csv'

        # Pull full run history via the W&B API (scan_history returns all rows)
        api = wandb.Api()
        run = api.run(f'{entity}/{project}/{run_id}')
        rows = list(run.scan_history())
        if not rows:
            return

        df = pd.DataFrame(rows)
        # Drop W&B internal columns
        df = df[[c for c in df.columns if not c.startswith('_')]]
        df.to_csv(out_path, index=False)
        print(f'[WandBCSVExportCallback] saved {len(df)} rows → {out_path}')
