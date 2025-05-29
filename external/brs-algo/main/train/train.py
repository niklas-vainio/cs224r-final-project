import hydra

import brs_algo.utils as U
from brs_algo.lightning import Trainer


@hydra.main(config_name="cfg", config_path="cfg", version_base="1.1")
def main(cfg):
    cfg.seed = U.set_seed(cfg.seed)
    trainer_ = Trainer(cfg)
    trainer_.trainer.loggers[-1].log_hyperparams(U.omegaconf_to_dict(cfg))
    trainer_.fit()


if __name__ == "__main__":
    main()
