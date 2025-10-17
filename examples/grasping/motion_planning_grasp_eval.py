import argparse
import pickle
import re
from importlib import metadata
from pathlib import Path

import torch

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
from rsl_rl.runners import OnPolicyRunner

from motion_planning_grasp_env import MotionPlanningGraspEnv


def load_policy(env, train_cfg, log_dir):
    """Load the trained reinforcement learning policy."""
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    # Find the latest checkpoint
    checkpoint_files = [
        f for f in log_dir.iterdir() if re.match(r"model_\d+\.pt", f.name)
    ]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")

    try:
        *_, last_ckpt = sorted(checkpoint_files)
    except ValueError as e:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}") from e

    runner.load(last_ckpt)
    print(f"Loaded checkpoint from {last_ckpt}")

    return runner.get_inference_policy(device=gs.device)


def evaluate_policy(env, policy, num_episodes=10):
    """
    Evaluate the policy and compute success metrics.

    Args:
        env: The environment
        policy: The trained policy
        num_episodes: Number of episodes to evaluate

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    episode_rewards = []
    phase_success_rates = {
        "pregrasp": [],
        "reach": [],
        "grasp": [],
        "lift": []
    }

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        with torch.no_grad():
            while not done:
                actions = policy(obs)
                obs, rewards, dones, infos = env.step(actions)
                episode_reward += rewards.mean().item()

                if dones.any():
                    done = True
                    # Record phase success rates
                    if "episode" in infos:
                        for phase_name in phase_success_rates.keys():
                            key = f"phase_{phase_name}_success_rate"
                            if key in infos["episode"]:
                                phase_success_rates[phase_name].append(infos["episode"][key])

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes}: Reward = {episode_reward:.2f}")

    # Compute statistics
    metrics = {
        "mean_reward": sum(episode_rewards) / len(episode_rewards),
        "std_reward": torch.tensor(episode_rewards).std().item(),
    }

    for phase_name, success_list in phase_success_rates.items():
        if success_list:
            metrics[f"{phase_name}_success_rate"] = sum(success_list) / len(success_list)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="motion_planning_grasp")
    parser.add_argument("--num_eval_episodes", type=int, default=10)
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record video during evaluation",
    )
    args = parser.parse_args()

    # Set PyTorch default dtype to float32
    torch.set_default_dtype(torch.float32)

    gs.init()

    log_dir = Path("logs") / args.exp_name

    # Load configurations
    if not (log_dir / "cfgs.pkl").exists():
        raise FileNotFoundError(f"Configuration file not found in {log_dir}. Please train the model first.")

    env_cfg, reward_cfg, robot_cfg, rl_train_cfg = pickle.load(
        open(log_dir / "cfgs.pkl", "rb")
    )

    # Modify config for evaluation
    env_cfg["num_envs"] = 10  # Use fewer environments for evaluation
    env_cfg["cube_collision"] = True  # Enable collision
    env_cfg["cube_fixed"] = False  # Object can move

    env = MotionPlanningGraspEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=True,  # Show viewer during evaluation
    )

    # Load the trained policy
    policy = load_policy(env, rl_train_cfg, log_dir)

    print("\n" + "="*60)
    print("Starting Evaluation")
    print("="*60)

    # Evaluate the policy
    metrics = evaluate_policy(env, policy, num_episodes=args.num_eval_episodes)

    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print("\nPhase Success Rates:")
    for phase in ["pregrasp", "reach", "grasp", "lift"]:
        key = f"{phase}_success_rate"
        if key in metrics:
            print(f"  {phase.capitalize()}: {metrics[key]:.2%}")
    print("="*60)

    # Run a single demonstration episode
    print("\nRunning demonstration episode...")
    obs, _ = env.reset()

    # Run for one full episode
    max_steps = int(env_cfg["episode_length_s"] / env_cfg["ctrl_dt"])

    with torch.no_grad():
        for step in range(max_steps):
            actions = policy(obs)
            obs, rewards, dones, infos = env.step(actions)

            # Print phase transitions
            if step % 50 == 0:
                current_phases = env.current_phase.cpu().numpy()
                phase_names = ["PRE-GRASP", "REACH", "GRASP", "LIFT"]
                phase_counts = {name: 0 for name in phase_names}
                for phase_id in current_phases:
                    if 0 <= phase_id < len(phase_names):
                        phase_counts[phase_names[phase_id]] += 1

                print(f"Step {step}: Phase distribution - {phase_counts}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

"""
Usage:

# Evaluate the trained model (default 10 episodes)
python examples/grasping/motion_planning_grasp_eval.py

# Evaluate with custom number of episodes
python examples/grasping/motion_planning_grasp_eval.py --num_eval_episodes 20

# Evaluate with custom experiment name
python examples/grasping/motion_planning_grasp_eval.py -e my_grasp_experiment

What to expect:
- The evaluation will run multiple episodes and compute average metrics
- Success rates for each phase will be displayed
- A demonstration episode will show the robot's behavior
- Phase distribution will be printed every 50 steps

Interpretation of results:
- Pre-grasp success rate: % of episodes where robot reaches above object
- Reach success rate: % of episodes where robot reaches down to object
- Grasp success rate: % of episodes where gripper closes on object
- Lift success rate: % of episodes where object is successfully lifted

Good performance:
- Pre-grasp: >80%
- Reach: >60%
- Grasp: >40%
- Lift: >20%

Metrics Reported:
Mean reward ± std
Success rate for each phase (pregrasp, reach, grasp, lift)
Phase distribution during executio
"""
