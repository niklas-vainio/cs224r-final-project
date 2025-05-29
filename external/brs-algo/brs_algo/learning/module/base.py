from typing import Any
from pytorch_lightning import LightningModule


class ImitationBaseModule(LightningModule):
    """
    Base class for IL algorithms that require 1) an environment, 2) a policy, and 3) rollout evaluation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, *args, **kwargs):
        self.on_train_start()
        loss, log_dict, batch_size = self.imitation_training_step(*args, **kwargs)
        log_dict = {f"train/{k}": v for k, v in log_dict.items()}
        log_dict["train/loss"] = loss
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, *args, **kwargs):
        loss, log_dict, real_batch_size = self.imitation_test_step(*args, **kwargs)
        log_dict = {f"val/{k}": v for k, v in log_dict.items()}
        log_dict["val/loss"] = loss
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=real_batch_size,
        )
        return log_dict

    def test_step(self, *args, **kwargs):
        loss, log_dict, real_batch_size = self.imitation_test_step(*args, **kwargs)
        log_dict = {f"test/{k}": v for k, v in log_dict.items()}
        log_dict["test/loss"] = loss
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=real_batch_size,
        )
        return log_dict

    def configure_optimizers(self):
        """
        Get optimizers, which are subsequently used to train.
        """
        raise NotImplementedError

    def imitation_training_step(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def imitation_test_step(self, *args, **kwargs) -> Any:
        raise NotImplementedError
