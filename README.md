# LeRobot Sim2Real

LeRobot Sim2real provides code to train with Reinforcement Learning in fast GPU parallelized simulation and rendering via [ManiSkill](https://github.com/haosulab/ManiSkill) and deploy to the real-world. The codebase is designed for use with the [🤗 LeRobot](https://github.com/huggingface/lerobot) library, which handles all of the hardware interfacing code. Once you clone and follow the installation instructions you can try out the [zero-shot RGB sim2real tutorial](./docs/zero_shot_rgb_sim2real.md) to train in pure simulation something that can pick up cubes in the real world like below:

https://github.com/user-attachments/assets/ca20d10e-d722-48fe-94af-f57e0b2b2fcd

Note that this project is still in a very early stage. There are many ways the sim2real can be improved (like more system ID tools, better reward functions etc.), but we plan to keep this repo extremely simple for readability and hackability.

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
conda activate ms3-lerobot
git clone https://github.com/StoneT2000/lerobot-sim2real.git
pip install -e .
pip install torch # install the version of torch that works for you
```

For sim2real transfer we also need to calibrate the real world camera. The default guide for this uses [EasyHEC](https://github.com/stonet2000/simple-easyhec) for that, which requires you to further run

```bash
pip install "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@729261dc64c4241ea36efda84fbf532cc8b425b8"
```

and setup SAM2 following their instructions: https://github.com/facebookresearch/sam2. You don't need this if you opt for the manual/slow camera calibration route.

The ManiSkill/SAPIEN simulator code is dependent on working NVIDIA drivers and vulkan packages. After running pip install above, if something is wrong with drivers/vulkan, please follow the troubleshooting guide here: https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#troubleshooting

To double check if the simulator is installed correctly, you can run

```
python -m mani_skill.examples.demo_random_action
```

Then we install lerobot which enable ease of use with all kinds of hardware.

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot && pip install -e .
```

Note that depending on what hardware you are using you might need to install additional packages in LeRobot. If you already installed lerobot somewhere else you can use that instead of running the command above.

## Sim2Real Tutorial

We currently provide a tutorial on how to train a RGB based model controlling an SO100 robot arm in simulation and deploying that zero-shot in the real world to grasp cubes. Follow the tutorial [here](./docs/zero_shot_rgb_sim2real.md). Note while SO101 looks similar to SO100, we have found that there are some key differences that make sim2real fail for SO101, we will update this repository once SO101 is modelled correctly.

The old version of the tutorial used some manual/slow camera calibration and can be found [here](./docs/zero_shot_rgb_sim2real_manual_calibration.md).

We are also working on a tutorial showing you how to make your own environments ready for sim2real, stay tuned!

## SO101 Support Branch Notes

- [x] Merge Viswesh-N pr [Added SO101 support - Tested with real robot](https://github.com/StoneT2000/lerobot-sim2real/pull/12/files)
- [ ] Take code changes from [SO-101 Support to Maniskill](https://github.com/haosulab/ManiSkill/pull/1171) and integrate into this repo, using Maniksill as a dependency.
  - [x] Copy the code chnages to this repo
  - [ ] Integrate and test the code changes
- [ ] Merge [Add EasyHEC camera extrinsics optimization for easier tutorial + calibration offsets guide ](https://github.com/StoneT2000/lerobot-sim2real/pull/17)
  - Simple EasyHEC gives us the camera instrinsic and extrinsic matrices to calibrate the camera. The environment will need to be updated to work with this format
- [ ] Test with SO101 arm and make adjustments and bug fixes as needed

## SO101 Testing Notes

- Run `python so101_sim2real_calibration_test.py`
  - This will launch Maniskill and allow you to control the robot in sim with by moving the real robot with torque disabled
  - You will probably need to update the camera config. Use `python -m lerobot.find_cameras opencv` to find your available camera indices, width, height, and FPS
