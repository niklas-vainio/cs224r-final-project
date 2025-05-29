from abc import ABC, abstractmethod

from pytorch_lightning import LightningModule


class BasePolicy(ABC, LightningModule):
    is_sequence_policy: bool = (
        False  # is this a feedforward policy or a policy requiring history
    )

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward the NN.
        """
        pass

    @abstractmethod
    def act(self, *args, **kwargs):
        """
        Given obs, return action.
        """
        pass


class BaseDiffusionPolicy(BasePolicy):
    @abstractmethod
    def get_optimizer_groups(self, *args, **kwargs):
        """
        Return a list of optimizer groups.
        """
        pass
