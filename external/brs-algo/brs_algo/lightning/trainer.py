from hydra.utils import instantiate

from brs_algo.lightning.lightning import LightingTrainer


class Trainer(LightingTrainer):
    def create_module(self, cfg):
        return instantiate(cfg.module, _recursive_=False)

    def create_data_module(self, cfg):
        return instantiate(cfg.data_module)

    def create_callbacks(self, cfg):
        is_rollout_eval = self.cfg.rollout_eval

        del_idxs = []
        for i, _cfg in enumerate(self.ckpt_cfg):
            eval_type = getattr(_cfg, "eval_type", None)
            if eval_type is not None:
                if is_rollout_eval and eval_type == "static":
                    del_idxs.append(i)
                elif not is_rollout_eval and eval_type == "rollout":
                    del_idxs.append(i)
                del _cfg["eval_type"]
        for i in reversed(del_idxs):
            del self.ckpt_cfg[i]
        return super().create_callbacks(cfg)
