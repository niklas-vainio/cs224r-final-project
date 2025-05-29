import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

import brs_algo.utils as U
from brs_algo.optim import default_optimizer_groups


class ObsTokenizer(nn.Module):
    def __init__(
        self,
        extractors: dict[str, nn.Module],
        *,
        use_modality_type_tokens: bool,
        token_dim: int,
        token_concat_order: list[str],
        strict: bool = True,
    ):
        assert set(extractors.keys()) == set(token_concat_order)
        super().__init__()
        self._extractors = nn.ModuleDict(extractors)
        self.output_dim = token_dim
        self._token_concat_order = token_concat_order
        self._strict = strict
        self._obs_groups = None
        self._use_modality_type_tokens = use_modality_type_tokens
        self._modality_type_tokens = None
        if use_modality_type_tokens:
            modality_type_tokens = {}
            for k in extractors:
                modality_type_tokens[k] = nn.Parameter(torch.zeros(token_dim))
            self._modality_type_tokens = nn.ParameterDict(modality_type_tokens)

    def forward(self, obs: dict[str, torch.Tensor]):
        """
        x: Dict of (B, T, ...)

        Each encoder should encode corresponding obs field to (B, T, E), where E = token_dim

        The final output interleaves encoded tokens along the time dimension
        """
        obs = self._group_obs(obs)
        self._check_obs_key_match(obs)
        x = {
            k: v.forward(obs[k]) for k, v in self._extractors.items()
        }  # dict of (B, T, E)
        if self._use_modality_type_tokens:
            for k in x:
                x[k] = x[k] + self._modality_type_tokens[k]
        x = rearrange(
            [x[k] for k in self._token_concat_order],
            "F B T E -> B (T F) E",
        )
        self._check_output_shape(obs, x)
        return x

    def _group_obs(self, obs):
        obs_keys = obs.keys()
        if self._obs_groups is None:
            # group by /
            obs_groups = {k.split("/")[0] for k in obs_keys}
            self._obs_groups = sorted(list(obs_groups))
        obs_rtn = {}
        for g in self._obs_groups:
            is_subgroup = any(k.startswith(f"{g}/") for k in obs_keys)
            if is_subgroup:
                obs_rtn[g] = {
                    k.split("/", 1)[1]: v
                    for k, v in obs.items()
                    if k.startswith(f"{g}/")
                }
            else:
                obs_rtn[g] = obs[g]
        return obs_rtn

    @U.call_once
    def _check_obs_key_match(self, obs: dict):
        if self._strict:
            assert set(self._extractors.keys()) == set(obs.keys())
        elif set(self._extractors.keys()) != set(obs.keys()):
            print(
                U.color_text(
                    f"[warning] obs key mismatch: {set(self._extractors.keys())} != {set(obs.keys())}",
                    "yellow",
                )
            )

    @U.call_once
    def _check_output_shape(self, obs, output):
        T = U.get_batch_size(U.any_slice(obs, np.s_[0]), strict=True)
        U.check_shape(
            output, (None, T * len(self._token_concat_order), self.output_dim)
        )

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        pg, pid = default_optimizer_groups(
            self,
            weight_decay=weight_decay,
            lr_scale=lr_scale,
            no_decay_filter=[
                "_extractors.*",
                "_modality_type_tokens.*",
            ],
        )
        return pg, pid

    @property
    def num_tokens_per_step(self):
        return len(self._token_concat_order)
