import argparse
import os
import pickle
from importlib import metadata
from pathlib import Path

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from motion_planning_grasp_env import MotionPlanningGraspEnv


def get_train_cfg(exp_name, max_iterations):
    """
    Training configuration for PPO reinforcement learning
    Configuration:
    Default: 4096 parallel environments
    500 training iterations
    PPO algorithm with adaptive learning rate
    Episode length: 5 seconds (500 timesteps at 100Hz)
"""
    rl_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,  # Slightly higher entropy for exploration
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0003,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "relu",
            "actor_hidden_dims": [256, 256, 128],
            "critic_hidden_dims": [256, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,  # Collect 24 steps per environment per iteration
        "save_interval": 100,  # Save checkpoint every 100 iterations
        "empirical_normalization": False,  # Disable observation normalization
        "seed": 1,
    }

    return rl_cfg_dict


def get_task_cfgs():
    """
    Environment and robot configuration for motion planning grasp task

    Task Description:
    - Sequential grasping task with 4 phases: pre-grasp, reach, grasp, lift
    - Object: 4cm cube at fixed position (0.65, 0.0, 0.02)
    - Robot: Franka Panda arm with 7 DOF arm + 2 DOF gripper
    - Control: 6-DOF end-effector control via inverse kinematics
    """
    env_cfg = {
        "num_envs": 4096,  # Number of parallel environments
        "num_obs": 18,  # Observation dimension (3+4+3+4+4)
        "num_actions": 6,  # Action dimension (6-DOF end-effector control)
        "action_scales": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],  # Max 5cm position, ~2.9° rotation per step
        "episode_length_s": 5.0,  # 5 seconds per episode
        "ctrl_dt": 0.01,  # 10ms control timestep (100Hz)
        "cube_size": [0.04, 0.04, 0.04],  # 4cm cube (like in tutorial)
        "cube_collision": True,  # Enable collision
        "cube_fixed": False,  # Object can be moved (not fixed)
    }

    # Reward function scales
    reward_scales = {
        "reach_target": 1.0,  # Dense reward for reaching phase targets
        "orientation": 0.5,  # Reward for maintaining downward orientation
        "lift_object": 2.0,  # Bonus for lifting the object
        "phase_completion": 1.0,  # Sparse reward for completing phases
        "action_smoothness": 0.1,  # Penalty for jerky motions
    }

    # Robot configuration (Franka Panda)
    robot_cfg = {
        "ee_link_name": "hand",
        "gripper_link_names": ["left_finger", "right_finger"],
        "default_arm_dof": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],  # Default joint angles
        "default_gripper_dof": [0.04, 0.04],  # Default gripper open position
        "ik_method": "dls_ik",  # Use Damped Least Squares IK
    }

    return env_cfg, reward_scales, robot_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="motion_planning_grasp")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=500)
    args = parser.parse_args()

    # === init Genesis ===
    gs.init(logging_level="warning", precision="32")

    # === task configs and training algo configs ===
    env_cfg, reward_scales, robot_cfg = get_task_cfgs()
    rl_train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # === log dir ===
    log_dir = Path("logs") / args.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save configurations
    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump((env_cfg, reward_scales, robot_cfg, rl_train_cfg), f)

    # === environment ===
    env_cfg["num_envs"] = args.num_envs
    env = MotionPlanningGraspEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_scales,
        robot_cfg=robot_cfg,
        show_viewer=args.vis,
    )

    # === runner ===
    runner = OnPolicyRunner(env, rl_train_cfg, log_dir, device=gs.device)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
Usage:

# Train with default settings (4096 envs, 500 iterations)
python examples/grasping/motion_planning_grasp_train.py

# Train with visualization (slower, for debugging)
python examples/grasping/motion_planning_grasp_train.py --vis

# Train with custom settings
python examples/grasping/motion_planning_grasp_train.py --num_envs 2048 --max_iterations 1000

# Train with custom experiment name
python examples/grasping/motion_planning_grasp_train.py -e my_grasp_experiment

Expected Training Time:
- With 4096 envs on GPU: ~30-60 minutes for 500 iterations
- Performance metrics to watch:
  * phase_pregrasp_success_rate: Should reach >0.8
  * phase_reach_success_rate: Should reach >0.6
  * phase_grasp_success_rate: Should reach >0.4
  * phase_lift_success_rate: Should reach >0.2
  * Mean reward: Should increase from ~0 to >50
"""
