import argparse
import os
import pickle
import shutil
import time
from datetime import datetime
from importlib import metadata

#  RL  Library: rsl-rl-lib (version 2.2.4 specifically required)
try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError(
        "Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'."
    ) from e
import genesis as gs

from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner


def log_train_time(start_time, end_time, args, log_dir):
    """Log training timing information to train_log.md"""
    duration = end_time - start_time

    # Prepare log entry
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"""
## Training Run - {timestamp}

**Experiment:** {args.exp_name}
**Number of Environments:** {args.num_envs}
**Max Iterations:** {args.max_iterations}
**Duration:** {duration:.2f} seconds ({duration/60:.2f} minutes) ({duration/3600:.2f} hours)
**Iterations/Second:** {args.max_iterations/duration:.4f}
**Seconds/Iteration:** {duration/args.max_iterations:.2f}

---
"""

    # Append to log file
    log_file = f"{log_dir}/train_log.md"
    with open(log_file, "a") as f:
        f.write(log_entry)

    print(f"Training timing logged to {log_file}")


def get_train_cfg(exp_name, max_iterations):
    #  RL Algorithm: PPO
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
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
        "num_steps_per_env": 24,  # Each simulated robot collects exactly 24 time steps ofexperience before the policy gets updated.
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 12, # 12 joints (3 per leg × 4 legs)
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        # PD
        "kp": 20.0,
        "kd": 0.5,
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    # Reward Configuration
    # Heavy Height Penalty (-50)
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument(
        "-B", "--num_envs", type=int, default=4096
    )  # 4096 identical Go2 robots are simulated simultaneously onthe GPU, each learning the same walking task independently.
    parser.add_argument("--max_iterations", type=int, default=101)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = Go2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    # Start timing the training
    train_start_time = time.time()
    print(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training {args.num_envs} environments for {args.max_iterations} iterations")

    runner.learn(
        num_learning_iterations=args.max_iterations, init_at_random_ep_len=True
    )

    # End timing and log results
    train_end_time = time.time()
    print(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        f"Total training time: {train_end_time - train_start_time:.2f} seconds ({(train_end_time - train_start_time)/60:.2f} minutes)"
    )

    # Log the timing information
    log_train_time(train_start_time, train_end_time, args, log_dir)


if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/go2_train.py
"""
