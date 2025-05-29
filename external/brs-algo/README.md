# BEHAVIOR Robot Suite: Streamlining Real-World Whole-Body Manipulation for Everyday Household Activities
<div align="center">

[Yunfan Jiang](https://yunfanj.com/),
[Ruohan Zhang](https://ai.stanford.edu/~zharu/),
[Josiah Wong](https://jdw.ong/),
[Chen Wang](https://www.chenwangjeremy.net/),
[Yanjie Ze](https://yanjieze.com/),
[Hang Yin](https://hang-yin.github.io/),
[Cem Gokmen](https://www.cemgokmen.com/),
[Shuran Song](https://shurans.github.io/),
[Jiajun Wu](https://jiajunwu.com/),
[Li Fei-Fei](https://profiles.stanford.edu/fei-fei-li)

<img src="media/SUSig-red.png" width=200>

[[Website]](https://behavior-robot-suite.github.io/)
[[arXiv]](https://arxiv.org/abs/2503.05652)
[[PDF]](https://behavior-robot-suite.github.io/assets/pdf/brs_paper.pdf)
[[Doc]](https://behavior-robot-suite.github.io/docs/)
[[Robot Code]](https://github.com/behavior-robot-suite/brs-ctrl)
[[Training Data]](https://huggingface.co/datasets/behavior-robot-suite/data)

[![Python Version](https://img.shields.io/badge/Python-3.11-blue.svg)](https://github.com/behavior-robot-suite/brs-algo)
[<img src="https://img.shields.io/badge/Framework-PyTorch-red.svg"/>](https://pytorch.org/)
[<img src="https://img.shields.io/badge/Doc-Passing-green.svg"/>](https://behavior-robot-suite.github.io/docs/)
[![GitHub license](https://img.shields.io/github/license/behavior-robot-suite/brs-algo)](https://github.com/behavior-robot-suite/brs-algo/blob/main/LICENSE)

![](media/pull.gif)
______________________________________________________________________
</div>

We introduce the **BEHAVIOR Robot Suite** (BRS), a comprehensive framework for learning whole-body manipulation to tackle diverse real-world household tasks. BRS addresses both hardware and learning challenges through two key innovations: **WB-VIMA** and [JoyLo](https://github.com/behavior-robot-suite/brs-ctrl).

WB-VIMA is an imitation learning algorithm designed to model whole-body actions by leveraging the robot’s inherent kinematic hierarchy. A key insight behind WB-VIMA is that robot joints exhibit strong interdependencies—small movements in upstream links (e.g., the torso) can lead to large displacements in downstream links (e.g., the end-effectors). To ensure precise coordination across all joints, WB-VIMA **conditions action predictions for downstream components on those of upstream components**, resulting in more synchronized whole-body movements. Additionally, WB-VIMA **dynamically aggregates multi-modal observations using self-attention**, allowing it to learn expressive policies while mitigating overfitting to proprioceptive inputs.

![](media/wbvima.gif)


## Getting Started

> [!TIP]
> 🚀 Check out the [doc](https://behavior-robot-suite.github.io/docs/sections/wbvima/overview.html) for detailed installation and usage instructions!

To train a WB-VIMA policy, simply run the following command:

```bash
python3 main/train/train.py data_dir=<HDF5_PATH> \
    bs=<BS> \
    arch=wbvima \
    task=<TASK_NAME> \
    exp_root_dir=<EXP_ROOT_DIR> \
    gpus=<NUM_GPUS> \
    use_wandb=<USE_WANDB> \
    wandb_project=<WANDB_PROJECT>
```

To deploy a WB-VIMA policy on the real robot, simply run the following command:

```bash
python3 main/rollout/<TASK_NAME>/rollout_async.py --ckpt_path <CKPT_PATH> --action_execute_start_idx <IDX>
```

## Check out Our Paper
Our paper is posted on [arXiv](https://arxiv.org/abs/2503.05652). If you find our work useful, please consider citing us! 

```bibtex
@article{jiang2025brs,
title = {BEHAVIOR Robot Suite: Streamlining Real-World Whole-Body Manipulation for Everyday Household Activities},
author = {Yunfan Jiang and Ruohan Zhang and Josiah Wong and Chen Wang and Yanjie Ze and Hang Yin and Cem Gokmen and Shuran Song and Jiajun Wu and Li Fei-Fei},
year = {2025},
journal = {arXiv preprint arXiv: 2503.05652}
}
```

## License
This codebase is released under the [MIT License](LICENSE).
