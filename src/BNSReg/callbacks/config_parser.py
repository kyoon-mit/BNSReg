from BNSReg.core.config import BNSDataModuleConfig

def __init__(self, **kwargs):
    cfg = BNSDataModuleConfig(**kwargs)
    super().__init__(cfg)
