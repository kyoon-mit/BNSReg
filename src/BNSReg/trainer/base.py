from lightning.pytorch import Trainer

class BNSTrainerBase(Trainer):
    def __init__(self):
        pass

    def train(self):
        pass
    
    def val(self):
        pass

    def test(self):
        pass