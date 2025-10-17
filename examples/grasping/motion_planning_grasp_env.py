import math
from typing import Literal

import genesis as gs
import torch
from genesis.utils.geom import transform_by_quat, transform_quat_by_quat, xyz_to_quat


class MotionPlanningGraspEnv:
    """
    Motion Planning Grasp Environment

    This environment teaches the robot to perform a complete grasp-and-lift task in a sequential manner:
    1. Pre-grasp: Move above the object (hand 25cm above cube)
    2. Reach: Move down to object (hand 13cm above ground)
    3. Grasp: Close gripper to grasp the object
    4. Lift: Lift the object to 28cm height

    ### Observation Space
    Observation Space (18D):
    3D position difference (finger to object)
    4D finger orientation (quaternion)
    3D object position
    4D object orientation (quaternion)
    4D one-hot encoding of current phase ← New addition
    obs_tensor = torch.cat([
        self.finger_pos - self.obj_pos,  # 3D position difference, Finger to object (x,y,z)
        self.finger_quat,                # 4D Finger orientation, Quaternion (w,x,y,z)
        self.obj_pos,                    # 3D object position, World coordinates (x,y,z)
        self.obj_quat,                   # 4D object orientation, Quaternion (w,x,y,z)
        self.phase_one_hot,              # 4D one-hot encoding of current phase
    ], dim=-1)  # Shape: (num_envs, 18)

    ### Action Space
    Action Space (6D):
    6-DOF end-effector control (3D position + 3D orientation deltas)
    Uses Damped Least Squares (DLS) inverse kinematics
    Scaled by 0.05 (max 5cm position, ~2.9° rotation per step)
    num_actions = 6
    # ├─ 0-2: End-effector position delta (dx, dy, dz) in meters
    # └─ 3-5: End-effector orientation delta (droll, dpitch, dyaw) in radians

    Control flow:
    Actions → scaled by 0.05
    Inverse kinematics (DLS solver) converts to joint positions
    PD controller drives motors to target joint positions
    """
    def __init__(
        self,
        env_cfg: dict,
        reward_cfg: dict,
        robot_cfg: dict,
        show_viewer: bool = False,
    ) -> None:
        self.num_envs = env_cfg["num_envs"]
        self.num_obs = env_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.device = gs.device

        self.ctrl_dt = env_cfg["ctrl_dt"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.ctrl_dt)

        # configs
        self.env_cfg = env_cfg
        self.reward_scales = reward_cfg
        self.action_scales = torch.tensor(env_cfg["action_scales"], device=self.device)

        # Phase tracking
        self.current_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.phase_time = torch.zeros(self.num_envs, device=self.device)

        # Phase constants
        self.PHASE_PREGRASP = 0
        self.PHASE_REACH = 1
        self.PHASE_GRASP = 2
        self.PHASE_LIFT = 3
        self.NUM_PHASES = 4

        # == setup scene ==
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                dt=self.ctrl_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(min(10, self.num_envs)))),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.ctrl_dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            show_viewer=show_viewer,
        )

        # == add ground ==
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # == add robot ==
        self.robot = Manipulator(
            num_envs=self.num_envs,
            scene=self.scene,
            args=robot_cfg,
            device=gs.device,
        )

        # == add object (cube) ==
        self.object = self.scene.add_entity(
            gs.morphs.Box(
                size=env_cfg["cube_size"],
                fixed=env_cfg["cube_fixed"],
                collision=env_cfg["cube_collision"],
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(0.0, 0.7, 1.0),  # Blue cube
                ),
            ),
        )

        # build
        self.scene.build(n_envs=env_cfg["num_envs"])
        # set pd gains (must be called after scene.build)
        self.robot.set_pd_gains()

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.ctrl_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros(
                (self.num_envs,), device=gs.device, dtype=gs.tc_float
            )

        # == init buffers ==
        self._init_buffers()
        self.reset()

    def _init_buffers(self) -> None:
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_int
        )
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=gs.device)
        self.extras = dict()
        self.extras["observations"] = dict()

        # Target positions for each phase
        self.pregrasp_height = 0.25  # 25cm above ground
        self.reach_height = 0.13  # 13cm above ground
        self.lift_height = 0.28  # 28cm above ground

        # Success tracking
        self.phase_success = torch.zeros(
            (self.num_envs, self.NUM_PHASES), dtype=torch.bool, device=self.device
        )

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        if len(envs_idx) == 0:
            return
        self.episode_length_buf[envs_idx] = 0
        self.current_phase[envs_idx] = self.PHASE_PREGRASP
        self.phase_time[envs_idx] = 0
        self.phase_success[envs_idx] = False

        # reset robot
        self.robot.reset(envs_idx)

        # reset object to fixed position (similar to tutorial)
        num_reset = len(envs_idx)
        # Object at (0.65, 0.0, 0.02) like in the tutorial
        obj_x = torch.ones(num_reset, device=self.device) * 0.65
        obj_y = torch.zeros(num_reset, device=self.device)
        obj_z = torch.ones(num_reset, device=self.device) * 0.02
        obj_pos = torch.stack([obj_x, obj_y, obj_z], dim=-1)

        # Downward facing quaternion
        obj_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(
            num_reset, 1
        )

        self.object.set_pos(obj_pos, envs_idx=envs_idx)
        self.object.set_quat(obj_quat, envs_idx=envs_idx)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        # Add phase completion statistics
        if len(envs_idx) > 0:
            for phase in range(self.NUM_PHASES):
                phase_name = ["pregrasp", "reach", "grasp", "lift"][phase]
                self.extras["episode"][f"phase_{phase_name}_success_rate"] = (
                    self.phase_success[envs_idx, phase].float().mean().item()
                )

    def reset(self) -> tuple[torch.Tensor, dict]:
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))

        obs, self.extras = self.get_observations()
        return obs, self.extras

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # update time
        self.episode_length_buf += 1
        self.phase_time += 1

        # apply action based on task
        actions = self.rescale_action(actions)

        # Determine gripper state based on phase
        open_gripper = (self.current_phase != self.PHASE_GRASP) & (self.current_phase != self.PHASE_LIFT)
        self.robot.apply_action(actions, open_gripper=open_gripper)
        self.scene.step()

        # Update phase transitions
        self._update_phases()

        # check termination
        env_reset_idx = self.is_episode_complete()
        if len(env_reset_idx) > 0:
            self.reset_idx(env_reset_idx)

        # compute reward based on task
        reward = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew

        # get observations and fill extras
        obs, self.extras = self.get_observations()

        return obs, reward, self.reset_buf, self.extras

    def _update_phases(self) -> None:
        """Update phase transitions based on task progress"""
        finger_pos = self.robot.center_finger_pose[:, :3]
        obj_pos = self.object.get_pos()
        obj_height = obj_pos[:, 2]

        # Phase 0: Pre-grasp (move above object at 25cm)
        pregrasp_mask = self.current_phase == self.PHASE_PREGRASP
        if pregrasp_mask.any():
            # Check if hand is above object at correct height
            target_pos = obj_pos.clone()
            target_pos[:, 2] = self.pregrasp_height
            dist_to_target = torch.norm(finger_pos - target_pos, dim=-1)

            # Transition to reach phase if close enough and stable
            success = (dist_to_target < 0.05) & (self.phase_time > 20)
            transition_mask = pregrasp_mask & success
            self.current_phase[transition_mask] = self.PHASE_REACH
            self.phase_time[transition_mask] = 0
            self.phase_success[transition_mask, self.PHASE_PREGRASP] = True

        # Phase 1: Reach (move down to object at 13cm)
        reach_mask = self.current_phase == self.PHASE_REACH
        if reach_mask.any():
            target_pos = obj_pos.clone()
            target_pos[:, 2] = self.reach_height
            dist_to_target = torch.norm(finger_pos - target_pos, dim=-1)

            success = (dist_to_target < 0.03) & (self.phase_time > 20)
            transition_mask = reach_mask & success
            self.current_phase[transition_mask] = self.PHASE_GRASP
            self.phase_time[transition_mask] = 0
            self.phase_success[transition_mask, self.PHASE_REACH] = True

        # Phase 2: Grasp (close gripper and ensure object is grasped)
        grasp_mask = self.current_phase == self.PHASE_GRASP
        if grasp_mask.any():
            # Check if object is still at original height (being held)
            # Give some time for gripper to close
            success = (self.phase_time > 30)
            transition_mask = grasp_mask & success
            self.current_phase[transition_mask] = self.PHASE_LIFT
            self.phase_time[transition_mask] = 0
            self.phase_success[transition_mask, self.PHASE_GRASP] = True

        # Phase 3: Lift (lift object to 28cm)
        lift_mask = self.current_phase == self.PHASE_LIFT
        if lift_mask.any():
            # Check if object is lifted
            success = (obj_height > 0.20) & (self.phase_time > 20)
            self.phase_success[lift_mask & success, self.PHASE_LIFT] = True

    def get_privileged_observations(self) -> None:
        return None

    def is_episode_complete(self) -> torch.Tensor:
        time_out_buf = self.episode_length_buf > self.max_episode_length

        # Episode ends on timeout
        self.reset_buf = time_out_buf

        # fill time out buffer for reward/value bootstrapping
        time_out_idx = (time_out_buf).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(
            self.reset_buf, device=gs.device, dtype=gs.tc_float
        )
        self.extras["time_outs"][time_out_idx] = 1.0
        return self.reset_buf.nonzero(as_tuple=True)[0]

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """
        RL Input: 18-dimensional state vector:
        - 3D position difference (finger to object)
        - 4D finger orientation quaternion
        - 3D object position
        - 4D object orientation quaternion
        - 4D one-hot encoding of current phase
        """
        # Current end-effector pose
        finger_pos, finger_quat = (
            self.robot.center_finger_pose[:, :3],
            self.robot.center_finger_pose[:, 3:7],
        )
        obj_pos, obj_quat = self.object.get_pos(), self.object.get_quat()

        # One-hot encoding of phase
        phase_one_hot = torch.zeros(
            (self.num_envs, self.NUM_PHASES), device=self.device
        )
        phase_one_hot[torch.arange(self.num_envs), self.current_phase] = 1.0

        obs_components = [
            finger_pos - obj_pos,  # 3D position difference
            finger_quat,  # current orientation (w, x, y, z)
            obj_pos,  # object position
            obj_quat,  # object orientation (w, x, y, z)
            phase_one_hot,  # 4D phase encoding
        ]
        obs_tensor = torch.cat(obs_components, dim=-1)
        self.extras["observations"]["critic"] = obs_tensor
        return obs_tensor, self.extras

    def rescale_action(self, action: torch.Tensor) -> torch.Tensor:
        rescaled_action = action * self.action_scales
        return rescaled_action

    # ------------ begin reward functions----------------
    """Reward Functions:
    reach_target: Dense reward for reaching phase-specific targets
    orientation: Reward for maintaining downward gripper orientation
    lift_object: Bonus reward for lifting the object
    phase_completion: Sparse reward for completing each phase
    action_smoothness: Penalty for jerky motions

    """
    def _reward_reach_target(self) -> torch.Tensor:
        """Reward for moving towards the current phase's target position"""
        finger_pos = self.robot.center_finger_pose[:, :3]
        obj_pos = self.object.get_pos()

        # Define target position based on current phase
        target_pos = obj_pos.clone()

        pregrasp_mask = self.current_phase == self.PHASE_PREGRASP
        reach_mask = self.current_phase == self.PHASE_REACH
        grasp_mask = self.current_phase == self.PHASE_GRASP
        lift_mask = self.current_phase == self.PHASE_LIFT

        target_pos[pregrasp_mask, 2] = self.pregrasp_height
        target_pos[reach_mask, 2] = self.reach_height
        target_pos[grasp_mask, 2] = self.reach_height  # Stay at reach height
        target_pos[lift_mask, 2] = self.lift_height

        # Distance to target
        dist = torch.norm(finger_pos - target_pos, dim=-1)

        # Exponential reward
        return torch.exp(-5.0 * dist)

    def _reward_orientation(self) -> torch.Tensor:
        """Reward for maintaining downward orientation (like in tutorial)"""
        finger_quat = self.robot.center_finger_pose[:, 3:7]
        target_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).repeat(
            self.num_envs, 1
        )

        # Quaternion distance (dot product)
        quat_dot = torch.abs(torch.sum(finger_quat * target_quat, dim=-1))
        quat_dot = torch.clamp(quat_dot, -1.0, 1.0)

        # Angle difference
        angle_diff = 2 * torch.acos(quat_dot)

        return torch.exp(-2.0 * angle_diff)

    def _reward_lift_object(self) -> torch.Tensor:
        """Bonus reward for successfully lifting the object"""
        obj_pos = self.object.get_pos()
        obj_height = obj_pos[:, 2]

        # Only reward in lift phase
        lift_mask = self.current_phase == self.PHASE_LIFT

        reward = torch.zeros(self.num_envs, device=self.device)
        reward[lift_mask] = torch.exp(5.0 * (obj_height[lift_mask] - 0.02))  # 0.02 is initial height

        return reward

    def _reward_phase_completion(self) -> torch.Tensor:
        """Sparse reward for completing each phase"""
        reward = torch.zeros(self.num_envs, device=self.device)

        # Give bonus when transitioning to next phase
        for phase in range(self.NUM_PHASES):
            # Check if phase was just completed (phase_time == 0 means just transitioned)
            phase_mask = (self.current_phase == phase + 1) & (self.phase_time == 1)
            reward[phase_mask] = 10.0

        # Extra bonus for completing final phase
        final_phase_mask = self.phase_success[:, self.PHASE_LIFT]
        reward[final_phase_mask] += 20.0

        return reward

    def _reward_action_smoothness(self) -> torch.Tensor:
        """Penalty for large actions (encourages smooth motion)"""
        # This is a negative reward (penalty)
        action_norm = torch.norm(self.robot._robot_entity.get_dofs_control_force(), dim=-1)
        return -0.001 * action_norm

    # ------------ end reward functions----------------


## ------------ robot ----------------
class Manipulator:
    def __init__(self, num_envs: int, scene: gs.Scene, args: dict, device: str = "cpu"):
        # == set members ==
        self._device = device
        self._scene = scene
        self._num_envs = num_envs
        self._args = args

        # == Genesis configurations ==
        material: gs.materials.Rigid = gs.materials.Rigid()
        morph: gs.morphs.URDF = gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
        )
        self._robot_entity: gs.Entity = scene.add_entity(material=material, morph=morph)

        self._gripper_open_dof = 0.04
        self._gripper_close_dof = 0.00

        self._ik_method: Literal["gs_ik", "dls_ik"] = args["ik_method"]

        # == some buffer initialization ==
        self._init()

    def set_pd_gains(self):
        # set control gains
        self._robot_entity.set_dofs_kp(
            torch.tensor([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
        )
        self._robot_entity.set_dofs_kv(
            torch.tensor([450, 450, 350, 350, 200, 200, 200, 10, 10]),
        )
        self._robot_entity.set_dofs_force_range(
            torch.tensor([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            torch.tensor([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )

    def _init(self):
        self._arm_dof_dim = self._robot_entity.n_dofs - 2  # total number of arm joints
        self._gripper_dim = 2  # number of gripper joints

        self._arm_dof_idx = torch.arange(self._arm_dof_dim, device=self._device)
        self._fingers_dof = torch.arange(
            self._arm_dof_dim,
            self._arm_dof_dim + self._gripper_dim,
            device=self._device,
        )
        self._left_finger_dof = self._fingers_dof[0]
        self._right_finger_dof = self._fingers_dof[1]
        self._ee_link = self._robot_entity.get_link(self._args["ee_link_name"])
        self._left_finger_link = self._robot_entity.get_link(
            self._args["gripper_link_names"][0]
        )
        self._right_finger_link = self._robot_entity.get_link(
            self._args["gripper_link_names"][1]
        )
        self._default_joint_angles = self._args["default_arm_dof"]
        if self._args["default_gripper_dof"] is not None:
            self._default_joint_angles += self._args["default_gripper_dof"]

    def reset(self, envs_idx: torch.IntTensor):
        if len(envs_idx) == 0:
            return
        self.reset_home(envs_idx)

    def reset_home(self, envs_idx: torch.IntTensor | None = None):
        if envs_idx is None:
            envs_idx = torch.arange(self._num_envs, device=self._device)
        default_joint_angles = torch.tensor(
            self._default_joint_angles, dtype=torch.float32, device=self._device
        ).repeat(len(envs_idx), 1)
        self._robot_entity.set_qpos(default_joint_angles, envs_idx=envs_idx)

    def apply_action(self, action: torch.Tensor, open_gripper: torch.Tensor) -> None:
        """
        Apply the action to the robot.
        open_gripper is now a boolean tensor of shape (num_envs,)
        """
        q_pos = self._robot_entity.get_qpos()
        if self._ik_method == "gs_ik":
            q_pos = self._gs_ik(action)
        elif self._ik_method == "dls_ik":
            q_pos = self._dls_ik(action)
        else:
            raise ValueError(f"Invalid control mode: {self._ik_method}")

        # Set gripper state per environment
        q_pos[open_gripper, self._left_finger_dof] = self._gripper_open_dof
        q_pos[open_gripper, self._right_finger_dof] = self._gripper_open_dof
        q_pos[~open_gripper, self._left_finger_dof] = self._gripper_close_dof
        q_pos[~open_gripper, self._right_finger_dof] = self._gripper_close_dof

        self._robot_entity.control_dofs_position(position=q_pos)

    def _gs_ik(self, action: torch.Tensor) -> torch.Tensor:
        """
        Genesis inverse kinematics
        """
        delta_position = action[:, :3]  # 3D position control
        delta_orientation = action[:, 3:6]  # 3D orientation control

        # compute target pose
        target_position = delta_position + self._ee_link.get_pos()
        quat_rel = xyz_to_quat(delta_orientation, rpy=True, degrees=False)
        target_orientation = transform_quat_by_quat(quat_rel, self._ee_link.get_quat())
        q_pos = self._robot_entity.inverse_kinematics(
            link=self._ee_link,
            pos=target_position,
            quat=target_orientation,
            dofs_idx_local=self._arm_dof_idx,
        )
        return q_pos

    def _dls_ik(self, action: torch.Tensor) -> torch.Tensor:
        """
        Damped least squares inverse kinematics
        """
        delta_pose = action[:, :6]  # 6-DOF action
        lambda_val = 0.01
        jacobian = self._robot_entity.get_jacobian(link=self._ee_link)
        jacobian_T = jacobian.transpose(1, 2)
        lambda_matrix = (lambda_val**2) * torch.eye(
            n=jacobian.shape[1], device=self._device
        )
        delta_joint_pos = (
            jacobian_T
            @ torch.inverse(jacobian @ jacobian_T + lambda_matrix)
            @ delta_pose.unsqueeze(-1)
        ).squeeze(-1)
        return self._robot_entity.get_qpos() + delta_joint_pos

    @property
    def base_pos(self):
        return self._robot_entity.get_pos()

    @property
    def ee_pose(self) -> torch.Tensor:
        """
        The end-effector pose (the hand pose)
        """
        pos, quat = self._ee_link.get_pos(), self._ee_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def left_finger_pose(self) -> torch.Tensor:
        pos, quat = self._left_finger_link.get_pos(), self._left_finger_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def right_finger_pose(self) -> torch.Tensor:
        pos, quat = (
            self._right_finger_link.get_pos(),
            self._right_finger_link.get_quat(),
        )
        return torch.cat([pos, quat], dim=-1)

    @property
    def center_finger_pose(self) -> torch.Tensor:
        """
        The center finger pose is the average of the left and right finger poses.
        """
        left_finger_pose = self.left_finger_pose
        right_finger_pose = self.right_finger_pose
        center_finger_pos = (left_finger_pose[:, :3] + right_finger_pose[:, :3]) / 2
        center_finger_quat = left_finger_pose[:, 3:7]
        return torch.cat([center_finger_pos, center_finger_quat], dim=-1)
