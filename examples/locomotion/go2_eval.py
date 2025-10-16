import argparse
import os
import pickle
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np

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

from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner


def log_eval_time(start_time, end_time, args, step_count, log_dir):
    """Log evaluation timing information to eval_log.md"""
    duration = end_time - start_time

    # Create log directory if it doesn't exist

    # Prepare log entry
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"""
    ## Evaluation Run - {timestamp}

    **Experiment:** {args.exp_name}
    **Checkpoint:** {args.ckpt}
    **Mode:** {'Headless' if args.headless else 'GUI'}
    **Video Recording:** {'Yes' if args.record_video else 'No'}
    **Steps Completed:** {step_count}
    **Duration:** {duration:.2f} seconds ({duration/60:.2f} minutes)
    **Steps/Second:** {step_count/duration:.2f}

    ---
    """

    # Append to log file
    log_file = f"{log_dir}/eval_log.md"
    with open(log_file, "a") as f:
        f.write(log_entry)

    print(f"Evaluation timing logged to {log_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode with camera recording",
    )
    parser.add_argument(
        "--record_video", action="store_true", help="Record video during evaluation"
    )
    parser.add_argument(
        "--video_filename",
        type=str,
        default="go2_eval.mp4",
        help="Output video filename",
    )
    parser.add_argument("--video_fps", type=int, default=30, help="Video FPS")
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # 1. Environment Setup
    # Initializes Genesis engine
    gs.init()
    log_dir = f"logs/{args.exp_name}"
    # Loads saved configuration from logs/{exp_name}/cfgs.pkl
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
        open(f"logs/{args.exp_name}/cfgs.pkl", "rb")
    )

    reward_cfg["reward_scales"] = {}

    # Creates Go2 environment with a single environment instance
    # Use headless mode if specified, otherwise show viewer
    show_viewer = not args.headless

    # Camera configuration for recording
    camera_config = None
    add_camera = args.headless or args.record_video
    if add_camera:
        camera_config = {
            "res": (1280, 720),  # HD resolution
            "pos": (3.5, 0.0, 2.5),  # Camera position behind and above the robot
            "lookat": (0, 0, 0.5),  # Look at the robot center
            "fov": 40,  # Field of view
            "GUI": False,  # No GUI window for headless mode
        }

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=show_viewer,
        add_camera=add_camera,
        camera_config=camera_config,
    )

    # ---------------------------------------------------------------

    # 2.  Model Loading
    # create an OnPolicyRunner instance
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    # load the trained policy from the specified checkpoint file.
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    #  Gets inference policy
    policy = runner.get_inference_policy(device=gs.device)

    # ---------------------------------------------------------------

    # 3. Evaluation Loop
    # Start timing the evaluation
    eval_start_time = time.time()
    print(f"Starting evaluation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Resets environment
    obs, _ = env.reset()

    # Start camera recording if enabled
    if env.camera is not None and args.record_video:
        print(f"Starting video recording to {args.video_filename}")
        env.camera.start_recording()

    step_count = 0
    max_steps = 1000  # Limit evaluation to 1000 steps for recording

    with torch.no_grad():
        while step_count < max_steps:
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)

            # Render camera if available (for recording or visualization)
            if env.camera is not None:
                # Dynamic camera movement - circle around the robot
                angle = step_count * 0.02  # Slow rotation
                camera_radius = 4.0
                env.camera.set_pose(
                    pos=(
                        camera_radius * np.cos(angle),
                        camera_radius * np.sin(angle),
                        2.5,
                    ),
                    lookat=(0, 0, 0.5),
                )
                env.camera.render()

            step_count += 1

            # Print progress every 100 steps
            if step_count % 100 == 0:
                print(f"Evaluation step: {step_count}/{max_steps}")

    # Stop recording and save video
    if env.camera is not None and args.record_video:
        print(f"Stopping recording and saving to {log_dir}/{args.video_filename}")
        env.camera.stop_recording(
            save_to_filename=f"{log_dir}/{args.video_filename}", fps=args.video_fps
        )
        print(f"Video saved successfully!")

    # End timing and log results
    eval_end_time = time.time()
    print(f"Evaluation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total evaluation time: {eval_end_time - eval_start_time:.2f} seconds")

    # Log the timing information
    log_eval_time(eval_start_time, eval_end_time, args, step_count, log_dir)


if __name__ == "__main__":
    main()


"""
# Standard evaluation with viewer
python examples/locomotion/go2_eval.py -e go2-walking --ckpt 100

# Headless evaluation with video recording
python examples/locomotion/go2_eval.py -e go2-walking --ckpt 100 --headless --record_video --video_filename log_dir/go2_eval.mp4

# Custom video settings
python examples/locomotion/go2_eval.py -e go2-walking --ckpt 100 --headless --record_video --video_filename my_video.mp4 --video_fps 60
"""
