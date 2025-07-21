#!/usr/bin/env python3
"""
SO101 Sim2Real Calibration Testing Script

This script connects to a real SO101 robot and mirrors its joint angles in a ManiSkill simulation.
Requirements:
1. SO101 robot should be powered on with torque disabled so you can move it by hand
2. Make sure the robot config has use_degrees=True
3. All motors should have use_degrees configuration enabled

Usage:
    python so101_sim2real_calibration_test.py

"""

import json
import time
import signal
import sys
from typing import Optional
import gymnasium as gym
import torch
import numpy as np
from lerobot_sim2real.utils.safety import setup_safe_exit
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from lerobot_sim2real.config.real_robot import create_real_robot
from mani_skill.agents.robots.lerobot.manipulator import LeRobotRealAgent
from mani_skill.envs.sim2real_env import Sim2RealEnv
import matplotlib.pyplot as plt
from dataclasses import dataclass
import threading


@dataclass
class CalibrationData:
    """Store calibration data for each joint"""
    rest_position: float = 0.0
    min_position: float = float('inf')
    max_position: float = float('-inf')
    direction_verified: bool = False


class SO101CalibrationTester:
    def __init__(self):
        self.real_robot = None
        self.real_agent = None
        self.sim_env = None
        self.real_env = None
        self.running = True
        self.joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        self.calibration_data = {joint: CalibrationData() for joint in self.joint_names}
        self.setup_signal_handlers()
        
    def setup_signal_handlers(self):
        """Setup signal handlers for safe shutdown"""
        def signal_handler(sig, frame):
            print("\nReceived interrupt signal. Shutting down safely...")
            self.running = False
            self.cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def connect_robot(self):
        """Connect to the real SO101 robot"""
        print("Connecting to SO101 robot...")
        self.real_robot = create_real_robot(uid="so101")
        self.real_robot.connect()
        
        # Disable torque so robot can be moved by hand
        print("Disabling torque - you can now move the robot by hand")
        self.real_robot.bus.disable_torque()
        
        # Verify use_degrees configuration
        print("\nVerifying motor configurations:")
        for motor_name, motor in self.real_robot.bus.motors.items():
            print(f"  {motor_name}: norm_mode = {motor.norm_mode}")
            if motor_name != "gripper" and motor.norm_mode.name != "DEGREES":
                print(f"WARNING: {motor_name} is not using DEGREES mode!")
        
        # Create LeRobotRealAgent with caching disabled for real-time updates
        self.real_agent = LeRobotRealAgent(self.real_robot, use_cached_qpos=False)
        print("Robot connected successfully")
    
    def setup_simulation(self):
        """Setup ManiSkill simulation environment"""
        print("Setting up simulation environment...")

        env_kwargs = dict(
            obs_mode="none",  # No observations needed
            control_mode="pd_joint_pos",
            sim_backend="physx_cpu",
            # sim_freq=100,
            # control_freq=20,
        )

        # Ensure SAPIEN uses OpenGL if Vulkan is unstable
        import os
        os.environ.setdefault("SAPIEN_DISABLE_VULKAN", "1")

        self.sim_env = gym.make(
            "SO101GraspCube-v1",  
            robot_uids="so101",
            render_mode="human",
            **env_kwargs,
        )

        self.sim_env.reset()

        sim_unwrapped = self.sim_env.unwrapped
        if hasattr(sim_unwrapped, "agent") and hasattr(sim_unwrapped.agent, "robot"):
            self.sim_robot = sim_unwrapped.agent.robot
        elif hasattr(sim_unwrapped, "robot"):
            self.sim_robot = sim_unwrapped.robot
        else:
            raise RuntimeError("Could not find robot handle in simulation environment")

        # Keep torque disabled on the real robot 
        self.real_robot.bus.disable_torque()

        print("Simulation environment ready")
    
    def read_real_robot_angles(self):
        """Read current joint angles from real robot using LeRobotRealAgent infrastructure"""
        try:
            qpos_tensor = self.real_agent.get_qpos()  
            qpos_rad = qpos_tensor.squeeze(0).cpu().numpy()  
            
            joint_positions = {}
            for i, joint_name in enumerate(self.joint_names):
                if i < len(qpos_rad):
                    joint_positions[joint_name] = np.rad2deg(qpos_rad[i])
            
            return joint_positions
        except Exception as e:
            print(f"Error reading robot angles: {e}")
            return None
    
    def set_sim_robot_angles(self, joint_positions):
        """Set simulation robot to match real robot angles"""
        try:
            if joint_positions is None:
                return
        
            qpos = []
            for joint_name in self.joint_names:
                if joint_name in joint_positions:
                    angle_deg = joint_positions[joint_name]
                    angle_rad = np.deg2rad(angle_deg)
                    qpos.append(angle_rad)
                else:
                    qpos.append(0.0)  

            qpos = torch.tensor(qpos, dtype=torch.float32).unsqueeze(0)
            self.sim_robot.set_qpos(qpos)
            
        except Exception as e:
            print(f"Error setting sim robot angles: {e}")
    
    def update_calibration_data(self, joint_positions):
        """Update min/max values for calibration tracking"""
        if joint_positions is None:
            return
            
        for joint_name in self.joint_names:
            if joint_name in joint_positions:
                angle = joint_positions[joint_name]
                calib_data = self.calibration_data[joint_name]
                calib_data.min_position = min(calib_data.min_position, angle)
                calib_data.max_position = max(calib_data.max_position, angle)
    
    def print_joint_angles(self, joint_positions):
        """Print current joint angles to terminal"""
        if joint_positions is None:
            return
            
        print("\nCurrent Joint Angles (degrees):")
        print("-" * 50)
        for joint_name in self.joint_names:
            if joint_name in joint_positions:
                angle = joint_positions[joint_name]
                calib_data = self.calibration_data[joint_name]
                min_val = calib_data.min_position if calib_data.min_position != float('inf') else 0
                max_val = calib_data.max_position if calib_data.max_position != float('-inf') else 0
                print(f"  {joint_name:12}: {angle:8.2f}° (range: {min_val:7.2f}° to {max_val:7.2f}°)")
        print("-" * 50)
    
    def print_calibration_instructions(self):
        """Print calibration instructions"""
        instructions = """
SO101 Sim2Real Calibration Instructions:
========================================

1. SETUP VERIFICATION:
   - Robot is powered on ✓
   - Torque is disabled (you can move robot by hand) ✓
   - use_degrees=True in robot config ✓
   - All motor norm_modes verified ✓

2. CALIBRATION PROCESS:
   a) Move robot to REST POSITION (joints parallel/perpendicular)
   b) Note the joint values - this is your '0' position reference
   c) Move each joint through its FULL RANGE OF MOTION
   d) Test each joint in BOTH DIRECTIONS
   e) Verify sim robot follows real robot EXACTLY

3. ALIGNMENT CHECKS:
   - Does sim robot move in SAME DIRECTION as real robot?
   - Are the joint limits correctly represented?
   - Does the rest position look correct in simulation?

4. CONTROLS:
   - Move real robot by hand
   - Watch sim robot follow in the GUI window
   - Monitor joint values in terminal
   - Press Ctrl+C to exit safely

Starting calibration test...
"""
        print(instructions)
    
    def run_calibration_loop(self):
        """Main calibration loop"""
        self.print_calibration_instructions()
        
        # Wait for user to be ready
        input("Press Enter when ready to start calibration testing...")
        
        print("\nStarting real-time joint mirroring...")
        print("Move the real robot and watch the simulation follow!")
        
        last_print_time = time.time()
        print_interval = 1.0  # Print every 1 second
        
        try:
            while self.running:
                loop_start = time.time()
                
                # Read real robot joint angles
                joint_positions = self.read_real_robot_angles()
                
                # Update simulation
                self.set_sim_robot_angles(joint_positions)
                
                # Update calibration data
                self.update_calibration_data(joint_positions)
                
                # Print joint angles periodically
                current_time = time.time()
                if current_time - last_print_time >= print_interval:
                    self.print_joint_angles(joint_positions)
                    last_print_time = current_time
                
                # Render simulation
                self.sim_env.render()
                
                # Control loop frequency (aim for ~30Hz)
                loop_time = time.time() - loop_start
                sleep_time = max(0, 1/30 - loop_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            print("\nCalibration test interrupted by user")
        except Exception as e:
            print(f"\nError during calibration: {e}")
        finally:
            self.cleanup()
    
    def print_final_calibration_report(self):
        """Print final calibration summary"""
        print("\n" + "="*60)
        print("FINAL CALIBRATION REPORT")
        print("="*60)
        
        for joint_name, calib_data in self.calibration_data.items():
            min_val = calib_data.min_position if calib_data.min_position != float('inf') else 0
            max_val = calib_data.max_position if calib_data.max_position != float('-inf') else 0
            range_val = max_val - min_val
            
            print(f"\n{joint_name.upper()}:")
            print(f"  Min Position: {min_val:8.2f}°")
            print(f"  Max Position: {max_val:8.2f}°")
            print(f"  Range:        {range_val:8.2f}°")
            print(f"  Rest Pos:     {calib_data.rest_position:8.2f}° (to be recorded)")
    
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        
        try:
            if self.real_robot:
                # Re-enable torque before disconnecting
                print("Re-enabling torque...")
                self.real_robot.bus.enable_torque()
                print("Disconnecting robot...")
                self.real_robot.disconnect()
        except Exception as e:
            print(f"Error during robot cleanup: {e}")
        
        try:
            if self.sim_env:
                self.sim_env.close()
        except Exception as e:
            print(f"Error during environment cleanup: {e}")
        
        self.print_final_calibration_report()
        print("✓ Cleanup completed")
    
    def run(self):
        """Run the complete calibration test"""
        try:
            print("SO101 Sim2Real Calibration Tester")

            self.connect_robot()
            self.setup_simulation()
            self.run_calibration_loop()

        except Exception as e:
            print(f"Error during calibration test: {e}")
            self.cleanup()


def main():
    """Main entry point"""
    tester = SO101CalibrationTester()
    tester.run()


if __name__ == "__main__":
    main() 