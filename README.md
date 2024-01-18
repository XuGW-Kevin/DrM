# DrM
DrM, a visual RL algorithm, minimizes the dormant ratio to guide exploration-exploitation trade-offs, achieving significant improvements in sample efficiency and asymptotic performance across diverse domains.

![image](images/title.gif)

# 🛠️ Installation Instructions
First, create a virtual environment and install all required packages. 
```bash
sudo apt update
sudo apt install libosmesa6-dev libegl1-mesa libgl1-mesa-glx libglfw3 
conda env create -f conda_env.yml 
conda activate drm
pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

Next, install the additional dependencies required for MetaWorld and Adroit. 
```
cd metaworld
pip install -e .
cd ..
cd mujoco-py
pip install -e .
cd ..
cd rrl-dependencies
pip install -e .
cd mj_envs
pip install -e .
cd ..
cd mjrl
pip install -e .
```

## 💻 Code Usage
If you run DrM on [DeepMind Control Suite](https://github.com/google-deepmind/dm_control), please use train_dmc.py to train DrM policies on different configs.

```bash
python train_dmc.py task=dog_walk agent=drm
```

If you run DrM on [MetaWorld](https://meta-world.github.io/), please use train_mw.py to train DrM policies on different configs.

```bash
python train_mw.py task=sweep-into agent=drm
python train_mw_sparse.py task=soccer agent=drm
```

If you run DrM on Adroit, please use train_adroit.py to train DrM policies on different configs.

```bash
python train_adroit.py task=pen agent=drm_adroit
```

## 📝 Citation

If you use our method or code in your research, please consider citing the paper as follows:

```
@inproceedings{
drm,
title={DrM: Mastering Visual Reinforcement Learning through Dormant Ratio Minimization},
author={Guowei Xu, Ruijie Zheng, Yongyuan Liang, Xiyao Wang, Zhecheng Yuan, Tianying Ji, Yu Luo, Xiaoyu Liu, Jiaxin Yuan, Pu Hua, Shuzhen Li, Yanjie Ze, Hal Daumé III, Furong Huang, Huazhe Xu.},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=MSe8YFbhUE}
}
```

## Acknowledgement
DrM is licensed under the MIT license. MuJoCo and DeepMind Control Suite are licensed under the Apache 2.0 license. We would like to thank DrQ-v2 authors for open-sourcing the [DrQv2](https://github.com/facebookresearch/drqv2) codebase. Our implementation builds on top of their repository.
