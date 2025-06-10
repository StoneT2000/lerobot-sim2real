# LeRobot Sim2Real
Training done in [ManiSkill](https://github.com/haosulab/ManiSkill). Deployment done in [🤗 LeRobot](https://github.com/huggingface/lerobot).


## Setup

Install this repo
```bash
conda create -n ms3-lerobot "python==3.11" # 3.11 is recommended
git clone https://github.com/StoneT2000/lerobot-sim2real.git
pip install -e .
pip install torch # install the version of torch that works for you
```

Then we install lerobot which enable ease of use with all kinds of hardware.

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot && pip install -e ".[dynamixel]" ## submodule lerobot?
```

For a quick test run

```bash
python lerobot_sim2real/scripts/random_actions.py
```

It will take random actions and save a video of the real environment

## Evaluation


```bash
python lerobot_sim2real/scripts/eval_agent.py --checkpoint <path_to_checkpoint>
# if no checkpoint is provided, it will use a random agent
```

## Supporting other Robots

TODO