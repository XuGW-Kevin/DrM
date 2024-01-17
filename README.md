# DrM
DrM, a visual RL algorithm, minimizes the dormant ratio to guide exploration-exploitation trade-offs, achieving significant improvements in sample efficiency and asymptotic performance across diverse domains.

![image](https://xugw-kevin.github.io/drm/asset/title.mp4)

## Installation
```bash
sudo apt update
sudo apt install libosmesa6-dev libegl1-mesa libgl1-mesa-glx libglfw3 
conda env create -f conda_env.yml 
conda activate drm
pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
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

## Usage
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

## Acknowledgement

This repo is based on https://github.com/facebookresearch/drqv2 .