'''
Smoke tests for import sanity.

These tests are intentionally boring.
If any of them fail, the project structure is broken.
'''


def test_package_import() -> None:
    import BNSReg  # noqa: F401


def test_cli_imports() -> None:
    from BNSReg.cli import train  # noqa: F401


def test_data_imports() -> None:
    from BNSReg.data import datasets  # noqa: F401
    from BNSReg.data import datamodule  # noqa: F401


def test_model_imports() -> None:
    from BNSReg.models import networks  # noqa: F401
    from BNSReg.models import lit_module  # noqa: F401


def test_callback_imports() -> None:
    from BNSReg.callbacks import csv_logger  # noqa: F401


def test_utils_imports() -> None:
    from BNSReg.utils import seed  # noqa: F401
