import shutil
import sys
from pathlib import Path

from lightning.pytorch.callbacks import Callback


class SaveConfigCallback(Callback):
    """Copies the --config YAML into the run's output directory.

    Fires on both fit_start and test_start so the config is saved
    even when only `test` is run.
    """

    def _run_dir(self, trainer) -> Path | None:
        if not (hasattr(trainer.logger, 'save_dir') and trainer.logger.save_dir):
            return None
        exp    = getattr(trainer.logger, 'experiment', None)
        run_id = (getattr(exp, 'id', None)
                  or getattr(trainer.logger, 'version', None)
                  or getattr(trainer.logger, 'id', None))
        if run_id is None:
            return None
        base    = Path(trainer.logger.save_dir)
        project = getattr(exp, 'project', None)
        if project:
            base = base / project
        return base / run_id

    def _copy(self, trainer):
        # Walk sys.argv to find --config <path>
        for i, arg in enumerate(sys.argv):
            if arg == '--config' and i + 1 < len(sys.argv):
                src = Path(sys.argv[i + 1])
                if not src.exists():
                    return
                run_dir = self._run_dir(trainer)
                if run_dir is None:
                    return
                run_dir.mkdir(parents=True, exist_ok=True)
                dst = run_dir / src.name
                shutil.copy2(src, dst)
                print(f'[SaveConfigCallback] {src.name} → {dst}')
                return

    def on_fit_start(self, trainer, pl_module):
        self._copy(trainer)

    def on_test_start(self, trainer, pl_module):
        self._copy(trainer)
