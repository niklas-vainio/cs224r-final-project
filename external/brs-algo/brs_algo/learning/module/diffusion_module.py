from typing import List, Union, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig

import brs_algo.utils as U
from brs_algo.optim import CosineScheduleFunction
from brs_algo.learning.policy.base import BasePolicy
from brs_algo.learning.module.base import ImitationBaseModule


class DiffusionModule(ImitationBaseModule):
    def __init__(
        self,
        *,
        # ====== policy ======
        policy: Union[BasePolicy, DictConfig],
        action_prediction_horizon: int,
        # ====== learning ======
        lr: float,
        use_cosine_lr: bool = False,
        lr_warmup_steps: Optional[int] = None,
        lr_cosine_steps: Optional[int] = None,
        lr_cosine_min: Optional[float] = None,
        lr_layer_decay: float = 1.0,
        weight_decay: float = 0.0,
        action_keys: List[str],
        loss_on_latest_obs_only: bool = False,
    ):
        super().__init__()
        if isinstance(policy, DictConfig):
            policy = instantiate(policy)
        self.policy = policy
        self._action_keys = action_keys
        self.action_prediction_horizon = action_prediction_horizon
        self.lr = lr
        self.use_cosine_lr = use_cosine_lr
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_cosine_steps = lr_cosine_steps
        self.lr_cosine_min = lr_cosine_min
        self.lr_layer_decay = lr_layer_decay
        self.weight_decay = weight_decay
        self.loss_on_latest_obs_only = loss_on_latest_obs_only

    def imitation_training_step(self, *args, **kwargs) -> Any:
        return self.imitation_training_step_seq_policy(*args, **kwargs)

    def imitation_test_step(self, *args, **kwargs):
        return self.imitation_val_step_seq_policy(*args, **kwargs)

    def imitation_training_step_seq_policy(self, batch, batch_idx):
        B = U.get_batch_size(
            U.any_slice(batch["action_chunks"], np.s_[0]),
            strict=True,
        )
        # obs data is dict of (N_chunks, B, window_size, ...)
        # action chunks is (N_chunks, B, window_size, action_prediction_horizon, A)
        # we loop over chunk dim
        main_data = U.unstack_sequence_fields(
            batch, batch_size=U.get_batch_size(batch, strict=True)
        )
        all_loss, all_mask_sum = [], 0
        for i, main_data_chunk in enumerate(main_data):
            # get padding mask
            pad_mask = main_data_chunk.pop(
                "pad_mask"
            )  # (B, window_size, L_pred_horizon)
            action_chunks = main_data_chunk.pop(
                "action_chunks"
            )  # dict of (B, window_size, L_pred_horizon, A)
            gt_actions = torch.cat(
                [action_chunks[k] for k in self._action_keys], dim=-1
            )
            transformer_output = self.policy(
                main_data_chunk
            )  # (B, L, E), where L is interleaved time and modality tokens
            loss = self.policy.compute_loss(
                transformer_output=transformer_output,
                gt_action=gt_actions,
            )  # (B, T_obs, T_act)
            if self.loss_on_latest_obs_only:
                mask = torch.zeros_like(pad_mask)
                mask[:, -1] = 1
                pad_mask = pad_mask * mask
            loss = loss * pad_mask
            all_loss.append(loss)
            all_mask_sum += pad_mask.sum()
        action_loss = torch.sum(torch.stack(all_loss)) / all_mask_sum
        # sum over action_prediction_horizon dim instead of avg
        action_loss = action_loss * self.action_prediction_horizon
        log_dict = {"diffusion_loss": action_loss}
        loss = action_loss
        return loss, log_dict, B

    def imitation_val_step_seq_policy(self, batch, batch_idx):
        """
        Will denoise as if it is in rollout
        but no env interaction
        """
        B = U.get_batch_size(
            U.any_slice(batch["action_chunks"], np.s_[0]),
            strict=True,
        )
        # obs data is dict of (N_chunks, B, window_size, ...)
        # action chunks is (N_chunks, B, window_size, action_prediction_horizon, A)
        # we loop over chunk dim
        main_data = U.unstack_sequence_fields(
            batch, batch_size=U.get_batch_size(batch, strict=True)
        )
        all_l1, all_mask_sum = {}, 0
        for i, main_data_chunk in enumerate(main_data):
            # get padding mask
            pad_mask = main_data_chunk.pop(
                "pad_mask"
            )  # (B, window_size, L_pred_horizon)
            gt_actions = main_data_chunk.pop(
                "action_chunks"
            )  # dict of (B, window_size, L_pred_horizon, A)
            transformer_output = self.policy(
                main_data_chunk
            )  # (B, L, E), where L is interleaved time and modality tokens
            pred_actions = self.policy.inference(
                transformer_output=transformer_output,
                return_last_timestep_only=False,
            )  # dict of (B, window_size, L_pred_horizon, A)
            for action_k in pred_actions:
                pred = pred_actions[action_k]
                gt = gt_actions[action_k]
                l1 = F.l1_loss(
                    pred, gt, reduction="none"
                )  # (B, window_size, L_pred_horizon, A)
                # sum over action dim
                l1 = l1.sum(dim=-1).reshape(
                    pad_mask.shape
                )  # (B, window_size, L_pred_horizon)
                if self.loss_on_latest_obs_only:
                    mask = torch.zeros_like(pad_mask)
                    mask[:, -1] = 1
                    pad_mask = pad_mask * mask
                l1 = l1 * pad_mask
                if action_k not in all_l1:
                    all_l1[action_k] = [
                        l1,
                    ]
                else:
                    all_l1[action_k].append(l1)
            all_mask_sum += pad_mask.sum()
        # avg on chunks dim, batch dim, and obs window dim so we can compare under different training settings
        all_l1 = {
            k: torch.sum(torch.stack(v)) / all_mask_sum for k, v in all_l1.items()
        }
        all_l1 = {k: v * self.action_prediction_horizon for k, v in all_l1.items()}
        summed_l1 = sum(all_l1.values())
        all_l1 = {f"l1_{k}": v for k, v in all_l1.items()}
        all_l1["l1"] = summed_l1
        return summed_l1, all_l1, B

    def configure_optimizers(self):
        optimizer_groups = self.policy.get_optimizer_groups(
            weight_decay=self.weight_decay,
            lr_layer_decay=self.lr_layer_decay,
            lr_scale=1.0,
        )

        optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        if self.use_cosine_lr:
            scheduler_kwargs = dict(
                base_value=1.0,  # anneal from the original LR value
                final_value=self.lr_cosine_min / self.lr,
                epochs=self.lr_cosine_steps,
                warmup_start_value=self.lr_cosine_min / self.lr,
                warmup_epochs=self.lr_warmup_steps,
                steps_per_epoch=1,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer,
                lr_lambda=CosineScheduleFunction(**scheduler_kwargs),
            )
            return (
                [optimizer],
                [{"scheduler": scheduler, "interval": "step"}],
            )

        return optimizer
