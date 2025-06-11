
import numpy as np
import torch
from lerobot_sim2real.config import real_robot


if __name__ == "__main__":
    real_robot = real_robot.create_real_robot(uid="s100")
    real_robot.connect()
    real_robot.bus.disable_torque()

    while True:
        qpos = real_robot.bus.sync_read("Present_Position")
        print("===")
        print(qpos)
        for k in qpos.keys():
            qpos[k] = np.deg2rad(qpos[k])
        print(qpos)
        input("enter...")

    real_robot.disconnect()