from lightning.pytorch import LightningModule

class LitBaseTask(LightningModule):
    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        # Callback: write loss, y_target, z_observed to file
        return loss