
from lerobot_sim2real.config import real_robot


if __name__ == "__main__":
    real_robot = real_robot.create_real_robot(uid="s100")
    real_robot.connect()
    real_robot.bus.disable_torque()

    while True:
        print(real_robot.get_observation())

    real_robot.disconnect()