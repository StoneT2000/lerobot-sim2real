from dataclasses import dataclass
from typing import Optional
import gymnasium as gym
import torch
import tyro
from lerobot_sim2real.config.real_robot import create_real_robot
from lerobot_sim2real.rl.agents.actor_critic import ActorCritic
from lerobot_sim2real.models.vision import NatureCNN

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
    real_robot = create_real_robot(uid="s100")
    real_robot.connect()
    # max control freq for lerobot really is just 60Hz
    real_agent = LeRobotRealAgent(real_robot)


    max_episode_steps = 200
    sim_env = gym.make(
        "SO100GraspCube-v1",
        obs_mode="rgb+segmentation",
        sim_config={"sim_freq": 120, "control_freq": 30},
        render_mode="sensors", # only sensors mode is supported right now for real envs, basically rendering the direct visual observations fed to policy
        max_episode_steps=max_episode_steps, # give our robot more time to try and re-try the task
        base_camera_settings=dict(
            pos=[0.69, 0.37, 0.28],
            fov=0.8256,
            target=[0.185, -0.15, 0.0]
        ),
        domain_randomization=False,
        greenscreen_overlay_path="greenscreen_background.png",
    )
    # you can apply most wrappers freely to the sim_env and the real env will use them
    sim_env = FlattenRGBDObservationWrapper(sim_env)
    sim_env = RecordEpisode(sim_env, output_dir="videos", save_trajectory=False, video_fps=sim_env.unwrapped.control_freq)
    
    real_env = Sim2RealEnv(sim_env=sim_env, agent=real_agent, obs_mode="rgb")
    sim_env.print_sim_details()
    sim_obs, _ = sim_env.reset(seed=2)
    real_obs, _ = real_env.reset()

    for k in sim_obs.keys():
        print(
            f"{k}: sim_obs shape: {sim_obs[k].shape}, real_obs shape: {real_obs[k].shape}"
        )

    # load agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = ActorCritic(sim_env, sample_obs=real_obs, feature_net=NatureCNN(real_obs))
    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded agent from {args.checkpoint}")
    else:
        print("No checkpoint provided, using random agent")

    pbar = tqdm(range(max_episode_steps))
    done = False

    # for plotting robot camera reads
    fig = plt.figure()
    ax = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    # Disable all default key bindings
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    fig.canvas.manager.key_press_handler_id = None

    use_sim_obs = False

    # initialize the plot
    overlaid_imgs, real_imgs, sim_imgs = overlay_envs(sim_env, real_env)
    im = ax.imshow(overlaid_imgs)
    im2 = ax2.imshow(sim_imgs)
    im3 = ax3.imshow(real_imgs)
    while not done:
        agent_obs = real_obs
        if use_sim_obs:
            agent_obs = FlattenRGBDObservationWrapper.observation(sim_env, sim_env.get_obs())
            agent_obs = {k: v.cpu() for k, v in agent_obs.items()}
            
        action = agent.get_action(agent_obs)
        if use_sim_obs:
            real_obs, _, terminated, truncated, info = sim_env.step(action)
        real_obs, _, terminated, truncated, info = real_env.step(action)
        input("enter")
        overlaid_imgs, real_imgs, sim_imgs = overlay_envs(sim_env, real_env)
        done = terminated or truncated
        # sim_env.render_human()
        im.set_data(overlaid_imgs)
        im2.set_data(sim_imgs)
        im3.set_data(real_imgs)
        # Redraw the plot
        fig.canvas.draw()
        fig.show()
        fig.canvas.flush_events()
        pbar.update(1)
    sim_env.close()
    real_env.close()

    print("Saved video to videos/0.mp4")

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)