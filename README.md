# LeRobot Sim2Real

LeRobot Sim2real provides code to train with Reinforcement Learning in fast GPU parallelized simulation and rendering via [ManiSkill](https://github.com/haosulab/ManiSkill) and deploy to the real-world. The codebase is designed for use with the [🤗 LeRobot](https://github.com/huggingface/lerobot) library, which handles all of the hardware interfacing code.


If you find this project useful, give this repo and [ManiSkill](https://github.com/haosulab/ManiSkill) a star! If you are using [SO100](https://github.com/TheRobotStudio/SO-ARM100/)/[LeRobot](https://github.com/huggingface/lerobot), make sure to also give them a star. If you use ManiSkill / this sim2real codebase in your research, please cite our [research paper](https://arxiv.org/abs/2410.00425):

```
@article{taomaniskill3,
  title={ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI},
  author={Stone Tao and Fanbo Xiang and Arth Shukla and Yuzhe Qin and Xander Hinrichsen and Xiaodi Yuan and Chen Bao and Xinsong Lin and Yulin Liu and Tse-kai Chan and Yuan Gao and Xuanlin Li and Tongzhou Mu and Nan Xiao and Arnav Gurha and Viswesh Nagaswamy Rajesh and Yong Woo Choi and Yen-Ru Chen and Zhiao Huang and Roberto Calandra and Rui Chen and Shan Luo and Hao Su},
  journal = {Robotics: Science and Systems},
  year={2025},
}
```

## Getting Started

Install this repo by running the following
```bash
conda create -n ms3-lerobot "python==3.11" # 3.11 is recommended
git clone https://github.com/StoneT2000/lerobot-sim2real.git
pip install -e .
pip install torch # install the version of torch that works for you
```

Then we install lerobot which enable ease of use with all kinds of hardware.

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot && pip install -e lerobot/
```

Note that depending on what hardware you are using you might need to install additional packages in LeRobot. If you already installed lerobot somewhere else you can use that instead of running the command above.

## Sim2Real Tutorial

We currently provide a tutorial on how to train a RGB based model controlling an SO100/SO101 robot arm in simulation and deploying that zero-shot in the real world to grasp cubes. Follow the tutorial [here](./docs/zero_shot_rgb_sim2real.md)