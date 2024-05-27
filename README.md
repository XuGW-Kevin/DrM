# DrM: Mastering Visual Reinforcement Learning through Dormant Ratio Minimization
<p align="center" style="font-size: 50px">
   <a href="https://arxiv.org/abs/2310.19668">[Paper]</a>&emsp;<a href="https://drm-rl.github.io/">[Project Website]</a>
</p>

This repository is the official PyTorch implementation of **DrM**. **DrM**, a visual reinforcement learning algorithm, minimizes the dormant ratio to guide exploration-exploitation trade-offs and achieves remarkable significant sample efficiency and asymptotic performance in the hardest locomotion and manipulation tasks.
<p align="center">
  <br><img src='images/title.gif' width="500"/><br>
</p>

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
cd rrl-dependencies
pip install -e .
cd mj_envs
pip install -e .
cd ..
cd mjrl
pip install -e .
```

Tips: please check that your mujoco_py can use gpu render to improve FPS during training.

```
mujoco_py.cymj
<module 'cymj' from './mujoco_py/generated/cymj_2.1.2.14_38_linuxgpuextensionbuilder_38.so'>
```

## 💻 Code Usage
If you would like to run DrM on [DeepMind Control Suite](https://github.com/google-deepmind/dm_control), please use train_dmc.py to train DrM policies on different configs.

```bash
python train_dmc.py task=dog_walk agent=drm
```

If you would like to run DrM on [MetaWorld](https://meta-world.github.io/), please use train_mw.py to train DrM policies on different configs.

```bash
python train_mw.py task=coffee-push agent=drm_mw
python train_mw.py task=disassemble agent=drm_mw
```

If you would like to run DrM on Adroit, please use train_adroit.py to train DrM policies on different configs.

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

## 🙏 Acknowledgement
DrM is licensed under the MIT license. MuJoCo and DeepMind Control Suite are licensed under the Apache 2.0 license. We would like to thank DrQ-v2 authors for open-sourcing the [DrQv2](https://github.com/facebookresearch/drqv2) codebase. Our implementation builds on top of their repository.
