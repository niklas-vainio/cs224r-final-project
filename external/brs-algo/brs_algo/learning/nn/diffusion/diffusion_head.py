from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers import SchedulerMixin
from einops import rearrange

from brs_algo.optim import default_optimizer_groups
from brs_algo.learning.nn.diffusion.unet import ConditionalUnet1D


class DiffusionHead(nn.Module):
    """
    Action head that generates actions through diffusion denosing.
    """

    """Backbone model of diffusion head, e.g., an MLP, a UNet, etc."""
    model: nn.Module
    """Action dimension."""
    action_dim: int
    """Action horizon."""
    action_horizon: int
    """Noise scheduler used in diffusion process."""
    noise_scheduler: SchedulerMixin
    """kwargs passed to noise scheduler's step method."""
    noise_scheduler_step_kwargs: Optional[dict] = None
    """Number of denoising steps during inference."""
    inference_denoise_steps: int

    def forward(
        self,
        obs: torch.Tensor,
        *,
        additional_input: Optional[torch.Tensor] = None,
        noisy_action: torch.Tensor,
        diffusion_timestep: torch.Tensor,
    ):
        """
        Run one pass to predict noise.

        Args:
            obs: Observation features of size (B, T_obs, D_obs), where T_obs = num_history_obs.
            additional_input: Additional input features of size (B, T_obs, D_additional).
            noisy_action: Noisy action of size (B, T_obs, T_act, A), where T_act = action prediction horizon.
                i.e., for each obs, the model predicts T_act future actions.
            diffusion_timestep: (B, T_obs, 1), timestep for diffusion process.

        Return:
            Predicted noise of size (B, T_obs, T_act, A).
        """
        assert (
            obs.ndim == 3
        ), f"obs should have 3 dimensions (B, T_obs, D_obs), got {obs.ndim}."
        if additional_input is not None:
            assert (
                additional_input.ndim == 3
            ), f"additional_input should have 3 dimensions (B, T_obs, D_additional), got {additional_input.ndim}."
            assert (
                additional_input.shape[:2] == obs.shape[:2]
            ), f"additional_input and obs should have the same batch size and time dimension."
            obs = torch.cat([obs, additional_input], dim=-1)
        assert (
            noisy_action.ndim == 4
        ), f"noisy_action should have 4 dimensions (B, T_obs, T_act, A), got {noisy_action.ndim}."
        assert (
            noisy_action.shape[:2] == obs.shape[:2]
        ), f"noisy_action and obs should have the same batch size and time dimension."
        flattened_noisy_action = rearrange(
            noisy_action, "B T_obs T_act A -> B T_obs (T_act A)"
        )
        denoise_in = torch.cat([obs, flattened_noisy_action], dim=-1)
        pred_eps = self.model(x=denoise_in, diffusion_t=diffusion_timestep)
        pred_eps = rearrange(
            pred_eps, "B T_obs (T_act A) -> B T_obs T_act A", T_act=self.action_horizon
        )
        return pred_eps

    def compute_loss(
        self,
        obs: torch.Tensor,
        *,
        additional_input: Optional[torch.Tensor] = None,
        gt_action: torch.Tensor,
    ):
        """
        Run one pass to predict noise and compute loss.

        Args:
            obs: Observation features of size (B, T_obs, D_obs), where T_obs = num_history_obs.
            additional_input: Additional input features of size (B, T_obs, D_additional).
            gt_action: Ground truth action of size (B, T_obs, T_act, A), where T_act = action prediction horizon.
                i.e., for each obs, the model predicts T_act future actions.
        """
        B, T_obs = obs.shape[:2]
        # flatten first two dim of gt_action
        gt_action = rearrange(gt_action, "B T_obs T_act A -> (B T_obs) T_act A")
        # sample noise
        noise = torch.randn(
            gt_action.shape, device=gt_action.device
        )  # (B * T_obs, T_act, A)
        # sample diffusion timesteps
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B * T_obs,),
            device=gt_action.device,
        )
        noisy_trajs = self.noise_scheduler.add_noise(
            gt_action, noise, timesteps
        )  # (B * T_obs, T_act, A)
        noisy_trajs = rearrange(
            noisy_trajs, "(B T_obs) T_act A -> B T_obs T_act A", B=B
        )
        noise = rearrange(noise, "(B T_obs) T_act A -> B T_obs T_act A", B=B)
        timesteps = rearrange(timesteps, "(B T_obs) -> B T_obs", B=B).unsqueeze(
            -1
        )  # (B, T_obs, 1)
        pred_eps = self.forward(
            obs,
            additional_input=additional_input,
            noisy_action=noisy_trajs,
            diffusion_timestep=timesteps,
        )  # (B, T_obs, T_act, A)
        mse_loss = F.mse_loss(pred_eps, noise, reduction="none")  # (B, T_obs, T_act, A)
        # sum over action dim instead of avg
        mse_loss = mse_loss.sum(dim=-1)  # (B, T_obs, T_act)
        return mse_loss

    @torch.no_grad()
    def inference(
        self,
        obs: torch.Tensor,
        *,
        additional_input: Optional[torch.Tensor] = None,
        return_last_timestep_only: bool = True,
    ):
        """
        Run inference to predict future actions.

        Args:
            obs: Observation features of size (B, T_obs, D_obs), where T_obs = num_history_obs.
            additional_input: Additional input features of size (B, T_obs, D_additional).
            return_last_timestep_only: Whether to return only the action chunks corresponding to the last timestep.
        """
        B, T_obs = obs.shape[:2]
        noisy_traj = torch.randn(
            size=(B, T_obs, self.action_horizon, self.action_dim),
            device=obs.device,
            dtype=obs.dtype,
        )
        if self.noise_scheduler.num_inference_steps != self.inference_denoise_steps:
            self.noise_scheduler.set_timesteps(self.inference_denoise_steps)

        for t in self.noise_scheduler.timesteps:
            timesteps = (
                torch.ones((B, T_obs, 1), device=obs.device, dtype=obs.dtype) * t
            )
            pred = self.forward(
                obs,
                additional_input=additional_input,
                noisy_action=noisy_traj,
                diffusion_timestep=timesteps,
            )  # (B, T_obs, T_act, A)
            # denosing
            pred = rearrange(pred, "B T_obs T_act A -> (B T_obs) T_act A")
            noisy_traj = rearrange(noisy_traj, "B T_obs T_act A -> (B T_obs) T_act A")
            noisy_traj = self.noise_scheduler.step(
                pred, t, noisy_traj, **self.noise_scheduler_step_kwargs
            ).prev_sample  # (B * T_obs, T_act, A)
            noisy_traj = rearrange(
                noisy_traj, "(B T_obs) T_act A -> B T_obs T_act A", B=B, T_obs=T_obs
            )
        if return_last_timestep_only:
            return noisy_traj[:, -1]
        return noisy_traj


class UNetDiffusionHead(DiffusionHead):
    def __init__(
        self,
        *,
        # ====== model ======
        obs_dim: int,
        action_dim: int,
        action_horizon: int,
        diffusion_step_embed_dim: int,
        unet_down_dims: List[int],
        unet_kernel_size: int,
        unet_n_groups: int,
        unet_cond_predict_scale: bool,
        # ====== noise scheduler ======
        noise_scheduler: SchedulerMixin,
        noise_scheduler_step_kwargs: Optional[dict] = None,
        # ====== inference ======
        inference_denoise_steps: int,
    ):
        super().__init__()
        self.model = ConditionalUnet1D(
            action_dim,
            local_cond_dim=None,
            global_cond_dim=obs_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=unet_down_dims,
            kernel_size=unet_kernel_size,
            n_groups=unet_n_groups,
            cond_predict_scale=unet_cond_predict_scale,
        )
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_step_kwargs = noise_scheduler_step_kwargs or {}
        self.inference_denoise_steps = inference_denoise_steps

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        pg, pid = default_optimizer_groups(
            self,
            weight_decay=weight_decay,
            lr_scale=lr_scale,
            no_decay_filter=[
                "model.*",
            ],
        )
        return pg, pid

    def forward(
        self,
        obs: torch.Tensor,
        *,
        additional_input: Optional[torch.Tensor] = None,
        noisy_action: torch.Tensor,
        diffusion_timestep: torch.Tensor,
    ):
        B, T_obs = obs.shape[:2]
        obs = rearrange(obs, "B T_obs D -> (B T_obs) D")
        noisy_action = rearrange(noisy_action, "B T_obs T_act A -> (B T_obs) T_act A")
        diffusion_timestep = rearrange(diffusion_timestep, "B T_obs 1 -> (B T_obs)")
        pred = self.model(
            sample=noisy_action,
            timestep=diffusion_timestep,
            global_cond=obs,
        )  # (B * T_obs, T_act, A)
        pred = rearrange(pred, "(B T_obs) T_act A -> B T_obs T_act A", B=B, T_obs=T_obs)
        return pred


class WholeBodyUNetDiffusionHead(nn.Module):
    def __init__(
        self,
        *,
        # ====== whole body ======
        whole_body_decoding_order: list[str],
        action_dim_per_part: dict[str, int],
        # ====== model ======
        obs_dim: int,
        action_horizon: int,
        diffusion_step_embed_dim: int,
        unet_down_dims: List[int],
        unet_kernel_size: int,
        unet_n_groups: int,
        unet_cond_predict_scale: bool,
        # ====== noise scheduler ======
        noise_scheduler: SchedulerMixin,
        noise_scheduler_step_kwargs: Optional[dict] = None,
        # ====== inference ======
        inference_denoise_steps: int,
    ):
        super().__init__()
        assert set(whole_body_decoding_order) == set(action_dim_per_part.keys())

        self.models = nn.ModuleDict()
        for i, part in enumerate(whole_body_decoding_order):
            additional_input_dim = 0
            for j in range(i):
                dependent_part = whole_body_decoding_order[j]
                additional_input_dim += (
                    action_dim_per_part[dependent_part] * action_horizon
                )
            model = ConditionalUnet1D(
                action_dim_per_part[part],
                local_cond_dim=None,
                global_cond_dim=obs_dim + additional_input_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=unet_down_dims,
                kernel_size=unet_kernel_size,
                n_groups=unet_n_groups,
                cond_predict_scale=unet_cond_predict_scale,
            )
            self.models[part] = model
        self.whole_body_decoding_order = whole_body_decoding_order
        self.action_dim_per_part = action_dim_per_part
        self.action_dim = sum(action_dim_per_part.values())
        self.action_horizon = action_horizon
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_step_kwargs = noise_scheduler_step_kwargs or {}
        self.inference_denoise_steps = inference_denoise_steps

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        pg, pid = default_optimizer_groups(
            self,
            weight_decay=weight_decay,
            lr_scale=lr_scale,
            no_decay_filter=[
                "models.*",
            ],
        )
        return pg, pid

    def forward(
        self,
        obs: torch.Tensor,
        *,
        dependent_action_input: dict[str, torch.Tensor],
        noisy_action: dict[str, torch.Tensor],
        diffusion_timestep: torch.Tensor | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Run one pass to predict noise.

        Args:
            obs: Observation features of size (B, T_obs, D_obs), where T_obs = num_history_obs.
            dependent_action_input: Dict of dependent action inputs, each of size (B, T_obs, T_act, A).
            noisy_action: Dict of noisy actions, each of size (B, T_obs, T_act, A).
            diffusion_timestep: (B, T_obs, 1), timestep for diffusion process. Can be the same for all parts,
             or different for each part (provide as dict).

        Return:
            Dict of predicted noise, each of size (B, T_obs, T_act, A), corresponding to each part.
        """
        assert (
            obs.ndim == 3
        ), f"obs should have 3 dimensions (B, T_obs, D_obs), got {obs.ndim}."
        B, T_obs = obs.shape[:2]
        assert set(dependent_action_input.keys()) == set(
            self.whole_body_decoding_order[:-1]
        )
        for dependent_part in self.whole_body_decoding_order[:-1]:
            assert dependent_action_input[dependent_part].shape == obs.shape[:2] + (
                self.action_horizon,
                self.action_dim_per_part[dependent_part],
            )
        assert set(noisy_action.keys()) == set(self.whole_body_decoding_order)
        for part in self.whole_body_decoding_order:
            assert noisy_action[part].shape == obs.shape[:2] + (
                self.action_horizon,
                self.action_dim_per_part[part],
            )
        if not isinstance(diffusion_timestep, dict):
            diffusion_timestep = {
                part: diffusion_timestep for part in self.whole_body_decoding_order
            }

        pred_eps_all_parts = {}
        for part_idx, part_name in enumerate(self.whole_body_decoding_order):
            all_dependent_action = None
            if part_idx > 0:
                all_dependent_action = []
                for j in range(part_idx):
                    dependent_action = dependent_action_input[
                        self.whole_body_decoding_order[j]
                    ]
                    dependent_action = rearrange(
                        dependent_action, "B T_obs T_act A -> B T_obs (T_act A)"
                    )
                    all_dependent_action.append(dependent_action)
                all_dependent_action = torch.cat(
                    all_dependent_action, dim=-1
                )  # (B, T_obs, D_dependent)

            denoise_in = rearrange(
                noisy_action[part_name], "B T_obs T_act A -> (B T_obs) T_act A"
            )
            if all_dependent_action is not None:
                global_cond = torch.cat(
                    [obs, all_dependent_action], dim=-1
                )  # (B, T_obs, D_obs + D_dependent)
            else:
                global_cond = obs
            global_cond = rearrange(global_cond, "B T_obs D -> (B T_obs) D")
            pred_eps = self.models[part_name](
                sample=denoise_in,
                timestep=rearrange(
                    diffusion_timestep[part_name], "B T_obs 1 -> (B T_obs)"
                ),
                global_cond=global_cond,
            )  # (B * T_obs, T_act, A)
            pred_eps = rearrange(
                pred_eps, "(B T_obs) T_act A -> B T_obs T_act A", B=B, T_obs=T_obs
            )
            pred_eps_all_parts[part_name] = pred_eps
        return pred_eps_all_parts

    def compute_loss(
        self,
        obs: torch.Tensor,
        *,
        gt_action: dict[str, torch.Tensor],
    ):
        """
        Run one pass to predict noise and compute loss.

        Args:
            obs: Observation features of size (B, T_obs, D_obs), where T_obs = num_history_obs.
            gt_action: dict of ground truth action of size (B, T_obs, T_act, A), where T_act = action prediction horizon.
                i.e., for each obs, the model predicts T_act future actions.
        """
        assert set(gt_action.keys()) == set(self.whole_body_decoding_order)
        B, T_obs = obs.shape[:2]

        noises, noisy_actions, diffusion_timesteps = {}, {}, {}
        for part in self.whole_body_decoding_order:
            # flatten first two dim of gt_action
            gt_action_this_part = rearrange(
                gt_action[part], "B T_obs T_act A -> (B T_obs) T_act A"
            )
            # sample noise
            noise_this_part = torch.randn(
                gt_action_this_part.shape, device=gt_action_this_part.device
            )  # (B * T_obs, T_act, A)
            # sample diffusion timesteps
            timesteps_this_part = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (B * T_obs,),
                device=gt_action_this_part.device,
            )
            noisy_trajs_this_part = self.noise_scheduler.add_noise(
                gt_action_this_part, noise_this_part, timesteps_this_part
            )  # (B * T_obs, T_act, A)
            noisy_trajs_this_part = rearrange(
                noisy_trajs_this_part, "(B T_obs) T_act A -> B T_obs T_act A", B=B
            )
            noise_this_part = rearrange(
                noise_this_part, "(B T_obs) T_act A -> B T_obs T_act A", B=B
            )
            timesteps_this_part = rearrange(
                timesteps_this_part, "(B T_obs) -> B T_obs", B=B
            ).unsqueeze(
                -1
            )  # (B, T_obs, 1)

            noises[part] = noise_this_part
            noisy_actions[part] = noisy_trajs_this_part
            diffusion_timesteps[part] = timesteps_this_part

        # for dependent action inputs, we use gt actions
        dependent_action_input = {
            part: gt_action[part] for part in self.whole_body_decoding_order[:-1]
        }

        pred_eps = self.forward(
            obs=obs,
            dependent_action_input=dependent_action_input,
            noisy_action=noisy_actions,
            diffusion_timestep=diffusion_timesteps,
        )  # dict of (B, T_obs, T_act, A)
        # concat all parts
        pred_eps = torch.cat(
            [pred_eps[part] for part in self.whole_body_decoding_order], dim=-1
        )
        noise = torch.cat(
            [noises[part] for part in self.whole_body_decoding_order], dim=-1
        )
        mse_loss = F.mse_loss(pred_eps, noise, reduction="none")  # (B, T_obs, T_act, A)
        # sum over action dim instead of avg
        mse_loss = mse_loss.sum(dim=-1)  # (B, T_obs, T_act)
        return mse_loss

    @torch.no_grad()
    def inference(
        self,
        obs: torch.Tensor,
        *,
        return_last_timestep_only: bool = True,
    ):
        """
        Run inference to predict future actions.

        Args:
            obs: Observation features of size (B, T_obs, D_obs), where T_obs = num_history_obs.
            return_last_timestep_only: Whether to return only the action chunks corresponding to the last timestep.
        """
        B, T_obs = obs.shape[:2]

        if self.noise_scheduler.num_inference_steps != self.inference_denoise_steps:
            self.noise_scheduler.set_timesteps(self.inference_denoise_steps)

        pred_action_all_parts = {}
        for part_idx, part in enumerate(self.whole_body_decoding_order):
            noisy_traj = torch.randn(
                size=(B, T_obs, self.action_horizon, self.action_dim_per_part[part]),
                device=obs.device,
                dtype=obs.dtype,
            )
            for t in self.noise_scheduler.timesteps:
                timesteps = (
                    torch.ones((B, T_obs, 1), device=obs.device, dtype=obs.dtype) * t
                )
                all_dependent_action = None
                if part_idx > 0:
                    all_dependent_action = []
                    for j in range(part_idx):
                        dependent_action = pred_action_all_parts[
                            self.whole_body_decoding_order[j]
                        ]
                        dependent_action = rearrange(
                            dependent_action, "B T_obs T_act A -> B T_obs (T_act A)"
                        )
                        all_dependent_action.append(dependent_action)
                    all_dependent_action = torch.cat(
                        all_dependent_action, dim=-1
                    )  # (B, T_obs, D_dependent)
                denoise_in = rearrange(
                    noisy_traj, "B T_obs T_act A -> (B T_obs) T_act A"
                )
                if all_dependent_action is not None:
                    global_cond = torch.cat([obs, all_dependent_action], dim=-1)
                else:
                    global_cond = obs
                global_cond = rearrange(global_cond, "B T_obs D -> (B T_obs) D")
                pred = self.models[part](
                    sample=denoise_in,
                    timestep=rearrange(timesteps, "B T_obs 1 -> (B T_obs)"),
                    global_cond=global_cond,
                )  # (B * T_obs, T_act, A)
                noisy_traj = rearrange(
                    noisy_traj, "B T_obs T_act A -> (B T_obs) T_act A"
                )
                noisy_traj = self.noise_scheduler.step(
                    pred, t, noisy_traj, **self.noise_scheduler_step_kwargs
                ).prev_sample  # (B * T_obs, T_act, A)
                noisy_traj = rearrange(
                    noisy_traj, "(B T_obs) T_act A -> B T_obs T_act A", B=B, T_obs=T_obs
                )
            pred_action_all_parts[part] = noisy_traj
        if return_last_timestep_only:
            return {k: v[:, -1] for k, v in pred_action_all_parts.items()}
        return pred_action_all_parts
