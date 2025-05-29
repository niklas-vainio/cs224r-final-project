from typing import List
import logging
import time
from copy import deepcopy

import sys
from omegaconf import DictConfig, OmegaConf, ListConfig
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.callbacks import (
    Callback,
    ProgressBar,
    TQDMProgressBar,
    RichProgressBar,
)
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.rank_zero import rank_zero_debug as rank_zero_debug_pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info as rank_zero_info_pl
from pytorch_lightning.callbacks import ModelCheckpoint

import brs_algo.utils as U


__all__ = [
    "LightingTrainer",
    "rank_zero_info",
    "rank_zero_debug",
    "rank_zero_warn",
    "rank_zero_info_pl",
    "rank_zero_debug_pl",
]

logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.WARNING)
logging.getLogger("torch.distributed.nn.jit.instantiator").setLevel(logging.WARNING)
U.logging_exclude_pattern(
    "root", patterns="*Reducer buckets have been rebuilt in this iteration*"
)


class LightingTrainer:
    def __init__(self, cfg: DictConfig, eval_only=False):
        """
        Args:
            eval_only: if True, will not save any model dir
        """
        cfg = deepcopy(cfg)
        OmegaConf.set_struct(cfg, False)
        self.cfg = cfg
        self.run_command_args = sys.argv[1:]
        U.register_omegaconf_resolvers()
        self.register_extra_resolvers(cfg)
        self._register_classes(cfg)
        # U.pprint_(OmegaConf.to_container(cfg, resolve=True))
        run_name = self.generate_run_name(cfg)
        self.run_dir = U.f_join(cfg.exp_root_dir, run_name)
        # rank_zero_info(OmegaConf.to_yaml(cfg, resolve=True))
        self._eval_only = eval_only
        self._resume_mode = None  # 'full state' or 'model only'
        if eval_only:
            rank_zero_info("Eval only, will not save any model dir")
        else:
            if "resume" in cfg and "ckpt_path" in cfg.resume and cfg.resume.ckpt_path:
                cfg.resume.ckpt_path = U.f_expand(
                    cfg.resume.ckpt_path.replace("_RUN_DIR_", self.run_dir).replace(
                        "_RUN_NAME_", run_name
                    )
                )
                self._resume_mode = (
                    "full state"
                    if cfg.resume.get("full_state", False)
                    else "model only"
                )
                rank_zero_info(
                    "=" * 80,
                    "=" * 80 + "\n",
                    f"Resume training from {cfg.resume.ckpt_path}",
                    f"\t({self._resume_mode})\n",
                    "=" * 80,
                    "=" * 80,
                    sep="\n",
                    end="\n\n",
                )
                time.sleep(3)
                assert U.f_exists(
                    cfg.resume.ckpt_path
                ), "resume ckpt_path does not exist"

            rank_zero_print("Run name:", run_name, "\nExp dir:", self.run_dir)
            U.f_mkdir(self.run_dir)
            U.f_mkdir(U.f_join(self.run_dir, "tb"))
            U.f_mkdir(U.f_join(self.run_dir, "logs"))
            U.f_mkdir(U.f_join(self.run_dir, "ckpt"))
            U.omegaconf_save(cfg, self.run_dir, "conf.yaml")
            rank_zero_print(
                "Checkpoint cfg:", U.omegaconf_to_dict(cfg.trainer.checkpoint)
            )
        self.cfg = cfg
        self.run_name = run_name
        self.ckpt_cfg = cfg.trainer.pop("checkpoint")
        self.data_module = self.create_data_module(cfg)
        self._monkey_patch_add_info(self.data_module)
        self.trainer = self.create_trainer(cfg)
        self.module = self.create_module(cfg)
        self.module.data_module = self.data_module
        self._monkey_patch_add_info(self.module)

        if not eval_only and self._resume_mode == "model only":
            ret = self.module.load_state_dict(
                U.torch_load(cfg.resume.ckpt_path)["state_dict"],
                strict=cfg.resume.strict,
            )
            U.rank_zero_warn("state_dict load status:", ret)

    def create_module(self, cfg) -> pl.LightningModule:
        return U.instantiate(cfg.module)

    def create_data_module(self, cfg) -> pl.LightningDataModule:
        return U.instantiate(cfg.data_module)

    def generate_run_name(self, cfg):
        return cfg.run_name + "_" + time.strftime("%Y%m%d-%H%M%S")

    def _monkey_patch_add_info(self, obj):
        """
        Add useful info to module and data_module so they can access directly
        """
        # our own info
        obj.run_config = self.cfg
        obj.run_name = self.run_name
        obj.run_command_args = self.run_command_args
        # add properties from trainer
        for attr in [
            "global_rank",
            "local_rank",
            "world_size",
            "num_nodes",
            "num_processes",
            "node_rank",
            "num_gpus",
            "data_parallel_device_ids",
        ]:
            if hasattr(obj, attr):
                continue
            setattr(
                obj.__class__,
                attr,
                # force capture 'attr'
                property(lambda self, attr=attr: getattr(self.trainer, attr)),
            )

    def create_loggers(self, cfg) -> List[pl.loggers.Logger]:
        if self._eval_only:
            loggers = []
        else:
            loggers = [
                pl_loggers.TensorBoardLogger(self.run_dir, name="tb", version=""),
                pl_loggers.CSVLogger(self.run_dir, name="logs", version=""),
            ]
        if cfg.use_wandb:
            loggers.append(
                pl_loggers.WandbLogger(
                    name=cfg.wandb_run_name, project=cfg.wandb_project, id=self.run_name
                )
            )
        return loggers

    def create_callbacks(self, cfg) -> List[Callback]:
        ModelCheckpoint.FILE_EXTENSION = ".pth"
        callbacks = []
        if isinstance(self.ckpt_cfg, DictConfig):
            ckpt = ModelCheckpoint(
                dirpath=U.f_join(self.run_dir, "ckpt"), **self.ckpt_cfg
            )
            callbacks.append(ckpt)
        else:
            assert isinstance(self.ckpt_cfg, ListConfig)
            for _cfg in self.ckpt_cfg:
                ckpt = ModelCheckpoint(dirpath=U.f_join(self.run_dir, "ckpt"), **_cfg)
                callbacks.append(ckpt)

        if "callbacks" in cfg.trainer:
            extra_callbacks = U.instantiate(cfg.trainer.pop("callbacks"))
            assert U.is_sequence(extra_callbacks), "callbacks must be a sequence"
            callbacks.extend(extra_callbacks)
        if not any(isinstance(c, ProgressBar) for c in callbacks):
            callbacks.append(CustomTQDMProgressBar())
        rank_zero_print(
            "Lightning callbacks:", [c.__class__.__name__ for c in callbacks]
        )
        return callbacks

    def create_trainer(self, cfg) -> pl.Trainer:
        assert "trainer" in cfg
        C = cfg.trainer
        # find_unused_parameters = C.pop("find_unused_parameters", False)
        # rank_zero_info("DDP Strategy", C.strategy)
        return U.instantiate(
            C, logger=self.create_loggers(cfg), callbacks=self.create_callbacks(cfg)
        )

    @property
    def tb_logger(self):
        return self.logger[0].experiment

    def fit(self):
        return self.trainer.fit(
            self.module,
            datamodule=self.data_module,
            ckpt_path=(
                self.cfg.resume.ckpt_path if self._resume_mode == "full state" else None
            ),
        )

    def validate(self):
        rank_zero_print("Start validation ...")
        assert "testing" in self.cfg, "`testing` sub-dict must be defined in config"
        ckpt_path = self.cfg.testing.ckpt_path
        if ckpt_path:
            ckpt_path = U.f_expand(ckpt_path)
            assert U.f_exists(ckpt_path), f"ckpt_path {ckpt_path} does not exist"
            rank_zero_info("Run validation on ckpt:", ckpt_path)
            ret = self.module.load_state_dict(
                U.torch_load(ckpt_path)["state_dict"], strict=self.cfg.testing.strict
            )
            U.rank_zero_warn("state_dict load status:", ret)
            ckpt_path = None  # not using pytorch lightning's load
        else:
            rank_zero_warn("WARNING: no ckpt_path specified, will NOT load any weights")
        return self.trainer.validate(
            self.module, datamodule=self.data_module, ckpt_path=ckpt_path
        )

    def _register_classes(self, cfg):
        U.register_callable("DDPStrategy", pl.strategies.DDPStrategy)
        U.register_callable("LearningRateMonitor", pl.callbacks.LearningRateMonitor)
        U.register_callable("ModelSummary", pl.callbacks.ModelSummary)
        U.register_callable("RichModelSummary", pl.callbacks.RichModelSummary)
        self.register_extra_classes(cfg)

    def register_extra_classes(self, cfg):
        pass

    def register_extra_resolvers(self, cfg):
        pass


@U.register_class
class CustomTQDMProgressBar(TQDMProgressBar):
    """
    https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.callbacks.ProgressBarBase.html#pytorch_lightning.callbacks.ProgressBarBase.get_metrics
    """

    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


# do the same overriding for RichProgressBar
@U.register_class
class CustomRichProgressBar(RichProgressBar):
    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


@rank_zero_only
def rank_zero_print(*msg, **kwargs):
    U.pprint_(*msg, **kwargs)


@rank_zero_only
def rank_zero_info(*msg, **kwargs):
    U.pprint_(
        U.color_text("[INFO]", color="green", styles=["reverse", "bold"]),
        *msg,
        **kwargs,
    )


@rank_zero_only
def rank_zero_warn(*msg, **kwargs):
    U.pprint_(
        U.color_text("[WARN]", color="yellow", styles=["reverse", "bold"]),
        *msg,
        **kwargs,
    )


@rank_zero_only
def rank_zero_debug(*msg, **kwargs):
    if rank_zero_debug.enabled:
        U.pprint_(
            U.color_text("[DEBUG]", color="blue", bg_color="on_grey"), *msg, **kwargs
        )


rank_zero_debug.enabled = True
