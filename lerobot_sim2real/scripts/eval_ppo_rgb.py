"""
This script is used to evaluate a random or RL trained agent on a real robot using the LeRobot system.
"""

from dataclasses import dataclass
import json
import random
from typing import Optional
import gymnasium as gym
import numpy as np
import torch
import tyro
from lerobot_sim2real.config.real_robot import create_real_robot
from lerobot_sim2real.rl.ppo_rgb import Agent

from lerobot_sim2real.utils.safety import setup_safe_exit
from mani_skill.agents.robots.lerobot.manipulator import LeRobotRealAgent
from mani_skill.envs.sim2real_env import Sim2RealEnv
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from tqdm import tqdm
from mani_skill.utils.visualization import tile_images
import matplotlib.pyplot as plt
@dataclass
class Args:
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to load agent weights from for evaluation. If None then a random agent will be used"""
    env_kwargs_json_path: Optional[str] = None
    """path to a json file containing additional environment kwargs to use. For real world evaluation this is not needed but if you want to turn on debug mode which visualizes the sim and real envs side by side you will need this"""
    debug: bool = False
    """if toggled, the sim and real envs will be visualized side by side"""
    continuous_eval: bool = True
    """If toggled, the evaluation will run until episode ends without user input. If false, at each timestep the user will be prompted to press enter to let the robot continue"""
    max_episode_steps: int = 100
    """The maximum number of control steps the real robot can take before we stop the episode and reset the environment. It is recommended to set this number to be larger than the value the sim env is set to, that way you can permit the
    robot more chances to recover from failures / solve the task."""
    num_episodes: Optional[int] = None
    """The number of episodes to evaluate for. If None, the evaluation will run until the user presses ctrl+c"""
    env_id: str = "SO100GraspCube-v1"
    """The environment id to use for evaluation. This should be the same as the environment id used for training."""
    seed: int = 1
    """seed of the experiment"""
    record_dir: Optional[str] = None
    """Directory to save recordings of the camera captured images. If none no recordings are saved"""
    control_freq: Optional[int] = 15
    """The control frequency of the real robot. For safety reasons we recommend setting this to 15Hz or lower as we permit the RL agent to take larger actions to move faster. If this is none, it will use the same control frequency the sim env uses."""

def overlay_envs(sim_env, real_env):
    """
    Overlays sim_env observtions onto real_env observations
    Requires matching ids between the two environments' sensors
    e.g. id=phone_camera sensor in real_env / real_robot config, must have identical id in sim_env
    """
    real_obs = real_env.get_obs()["sensor_data"]
    sim_obs = sim_env.get_obs()["sensor_data"]
    assert sorted(real_obs.keys()) == sorted(
        sim_obs.keys()
    ), f"real camera names {real_obs.keys()} and sim camera names {sim_obs.keys()} differ"

    overlaid_dict = sim_env.get_obs()["sensor_data"]
    overlaid_imgs = []
    for name in overlaid_dict:
        real_imgs = real_obs[name]["rgb"][0] / 255
        sim_imgs = overlaid_dict[name]["rgb"][0].cpu() / 255
        overlaid_imgs.append(0.5 * real_imgs + 0.5 * sim_imgs)

    return tile_images(overlaid_imgs), real_imgs, sim_imgs

def main(args: Args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ### Create and connect the real robot, wrap it to make it interfaceable with ManiSkill sim2real environments ###    
    real_robot = create_real_robot(uid="so100")
    real_robot.connect()
    real_agent = LeRobotRealAgent(real_robot)

    ### Setup the sim environment to make various checks for sim2real alignment and debugging possible ###
    env_kwargs = dict(
        obs_mode="rgb+segmentation",
        render_mode="sensors", # only sensors mode is supported right now for real envs, basically rendering the direct visual observations fed to policy
        max_episode_steps=args.max_episode_steps, # give our robot more time to try and re-try the task
        domain_randomization=False,
        reward_mode="none"
    )
    if args.control_freq is not None:
        env_kwargs["sim_config"] = {"control_freq": args.control_freq}
    if args.env_kwargs_json_path is not None:
        with open(args.env_kwargs_json_path, "r") as f:
            env_kwargs.update(json.load(f))
    
    sim_env = gym.make(
        args.env_id,
        **env_kwargs
    )
    # you can apply most wrappers freely to the sim_env and the real_env will use them as well
    sim_env = FlattenRGBDObservationWrapper(sim_env)
    if args.record_dir is not None:
        # TODO (stao): verify this wrapper works
        sim_env = RecordEpisode(sim_env, output_dir=args.record_dir, save_trajectory=False, video_fps=sim_env.unwrapped.control_freq)
    
    # The Sim2RealEnv class uses the sim_env to help make various checks for sim2real alignment (e.g. observation space is the same, cameras are the similar)
    # and will always try its best to apply all wrappers you used on the sim env to the real env as well.
    real_env = Sim2RealEnv(sim_env=sim_env, agent=real_agent)
    # sim_env.print_sim_details()
    sim_obs, _ = sim_env.reset()
    real_obs, _ = real_env.reset()

    for k in sim_obs.keys():
        print(
            f"{k}: sim_obs shape: {sim_obs[k].shape}, real_obs shape: {real_obs[k].shape}"
        )

    
    ### Safety setups. Close environments/turn off robot upon ctrl+c ###
    setup_safe_exit(sim_env, real_env, real_agent)
        

    ### Load our checkpoint ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(sim_env, sample_obs=real_obs)
    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded agent from {args.checkpoint}")
    else:
        print("No checkpoint provided, using random agent")
    agent.to(device)

    
    ### Visualization setup for debug modes ###
    if args.debug:
        fig = plt.figure()
        ax = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3)

        # Disable all default key bindings
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        fig.canvas.manager.key_press_handler_id = None


        # initialize the plot
        overlaid_imgs, real_imgs, sim_imgs = overlay_envs(sim_env, real_env)
        im = ax.imshow(overlaid_imgs)
        im2 = ax2.imshow(sim_imgs)
        im3 = ax3.imshow(real_imgs)

    ### Main evaluation loop ###
    episode_count = 0
    while args.num_episodes is None or episode_count < args.num_episodes:
        print(f"Evaluation Episode {episode_count}")
        for _ in tqdm(range(args.max_episode_steps)):
            agent_obs = real_obs

            agent_obs = {k: v.to(device) for k, v in agent_obs.items()}
            action = agent.get_action(agent_obs)
            if not args.continuous_eval:
                input("Press enter to continue to next timestep")
            real_obs, _, terminated, truncated, info = real_env.step(action.cpu().numpy())
            
            if args.debug:
                overlaid_imgs, real_imgs, sim_imgs = overlay_envs(sim_env, real_env)
                im.set_data(overlaid_imgs)
                im2.set_data(sim_imgs)
                im3.set_data(real_imgs)
                # Redraw the plot
                fig.canvas.draw()
                fig.show()
                fig.canvas.flush_events()

        episode_count += 1
        real_env.reset()
    sim_env.close()
    real_env.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)