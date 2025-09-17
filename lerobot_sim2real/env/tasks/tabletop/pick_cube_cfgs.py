"""
PickCube-v1 is a basic/common task which defaults to using the panda robot. It is also used as a testing task to check whether a robot with manipulation
capabilities can be simulated and trained properly. The configs below set the pick cube task differently to ensure the cube is within reach of the robot tested
and the camera angles are reasonable.
"""

PICK_CUBE_CONFIGS = {
    "so100": {
        "cube_half_size": 0.0125,
        "goal_thresh": 0.0125 * 1.25,
        "cube_spawn_half_size": 0.05,
        "cube_spawn_center": (-0.46, 0),
        "max_goal_height": 0.08,
        "sensor_cam_eye_pos": [-0.27, 0, 0.4],
        "sensor_cam_target_pos": [-0.56, 0, -0.25],
        "human_cam_eye_pos": [-0.1, 0.3, 0.4],
        "human_cam_target_pos": [-0.46, 0.0, 0.1],
    },
    "so101": {
        "cube_half_size": 0.0125,
        "goal_thresh": 0.0125 * 1.25,
        "cube_spawn_half_size": 0.05,
        "cube_spawn_center": (-0.46, 0),
        "max_goal_height": 0.08,
        "sensor_cam_eye_pos": [-0.27, 0, 0.4],
        "sensor_cam_target_pos": [-0.56, 0, -0.25],
        "human_cam_eye_pos": [-0.1, 0.3, 0.4],
        "human_cam_target_pos": [-0.46, 0.0, 0.1],
    },
}
