import genesis as gs
import numpy as np

"""

What is Inverse Kinematics?
- Inverse kinematics is a mathematical technique to calculate the joint angles needed to position a robot's end-effector (like a gripper or hand) at a desired location and orientation in 3D space.

Forward Kinematics (FK): Given joint angles → Calculate where the end-effector ends up
Inverse Kinematics (IK): Given desired end-effector pose → Calculate what joint angles are needed


- For IK solving, you simply tell the robot’s IK solver which link is the end-effector, and specify the target pose.

- Use cases of IK solving in the current example:
  - Pre-grasp position: Hand 25cm above the cube
  - Reach position: Hand just above cube at 13cm height
  - Lift position: Hand lifted to 28cm height


What is Motion Planning?

Motion planning is the process of finding a collision-free path for a robot to move from its current configuration to a goal configuration. 
It's like GPS navigation, but for robots moving through joint space.

Given: Current joint positions and goal joint positions
Find: A smooth sequence of intermediate joint positions (waypoints) that:
    - Connects start to goal
    - Avoids collisions with obstacles and self-collisions
    - Respects joint limits and velocity constraints
    - Is smooth and efficient

"""


############################## init #######################################
gs.init(backend=gs.gpu)


###########################################################################
# SCENE SETUP
###########################################################################

##########################  Scene Initialization ##########################
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5), # Camera 3m forward, 1m left, 1.5m up
        camera_lookat=(0.0, 0.0, 0.5), # Looking at table height
        camera_fov=30, # Narrow field of view
        max_FPS=60, # 60Hz rendering
    ),
    sim_options=gs.options.SimOptions(
        dt=0.01,  # 10ms timestep (100Hz physics)
    ),
    show_viewer=False,   # Headless mode (no GUI)
    rigid_options=gs.options.RigidOptions(
        enable_collision=True,  # Enable collision detection
    ),
)

########################## Entity Creation ###############################
# (1). Ground plane
plane = scene.add_entity(
    gs.morphs.Plane(),
)
# (2). Target object: 4cm cube at position (0.65, 0.0, 0.02)
cube = scene.add_entity(
    gs.morphs.Box(
        size=(0.04, 0.04, 0.04),
        pos=(0.65, 0.0, 0.02),
    )
)
# (3). Franka Panda robot
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)

###########################################################################
# ROBOT CONFIGURATION
###########################################################################


# (1). Degrees of Freedom (DOF) : 9 degrees of freedom for Franka robot
# Total: 9 DOFs
# ├─ Arm: 7 revolute joints
# │  ├─ Joint 0: Shoulder pan (rotates around vertical axis)
# │  ├─ Joint 1: Shoulder lift (raises/lowers arm)
# │  ├─ Joint 2: Upper arm roll (twists upper arm)
# │  ├─ Joint 3: Elbow flex (bends elbow)
# │  ├─ Joint 4: Forearm roll (twists forearm)
# │  ├─ Joint 5: Wrist flex (bends wrist)
# │  └─ Joint 6: Wrist roll (twists hand)
# └─ Gripper: 2 prismatic joints
#    ├─ Joint 7: Left finger (slides in/out)
#    └─ Joint 8: Right finger (slides in/out)
motors_dof = np.arange(7) # DOFs 0-6 (arm joints) - 7 joints
fingers_dof = np.arange(7, 9) # # DOFs 7-8 (gripper fingers) - 2 fingers


# (2). PD Control Gains
# Note: the following values are tuned for achieving best behavior with Franka
# Typically, each new robot would have a different set of parameters.
# Sometimes high-quality URDF or XML file would also provide this and will be parsed.

#  Proportional gains (Kp) - position error response
franka.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
) # First 7 values: Arm joint gains (higher values); Last 2 values: Gripper finger gains (lower values)

# Derivative gains (Kv) - velocity damping
franka.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
)

# Force limits (Nm for revolute, N for prismatic)
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
)

# (3). Video Recording Setup
camera = scene.add_camera(
    res=(1280, 720),
    pos=(3, -1, 1.5),
    lookat=(0.0, 0.0, 0.5),
    fov=30,
    GUI=False,
)
################################### build ################################
scene.build()

# Start video recording
print("Starting video recording...")
camera.start_recording()

# ------------------------------------------------------------------------
# THE GRASPING SEQUENCE STARTS
# ------------------------------------------------------------------------


###########################################################################
# PHASE 1: PRE-GRASP (Move Above Object)
###########################################################################

# (1). Step 1: Compute Target Joint Angles with IK

# get the end-effector link
end_effector = franka.get_link("hand")

# move to pre-grasp pose
qpos = franka.inverse_kinematics( # Use inverse kinetics (IK) to solve the joint position given a target end-effector pose
    link=end_effector,  # Which link to position (the "hand")
    pos=np.array([0.65, 0.0, 0.25]), # Target position: 25cm above cube
    quat=np.array([0, 1, 0, 0]), #  Target orientation: Gripper pointing downward
)

# set gripper to open position
qpos[-2:] = 0.04  # Last 2 DOFs: finger positions (4cm = fully open)

# (2). Step 2: Motion Planning

# 2.1 plan path: # Plan a smooth path from current pose to target pose 
path = franka.plan_path(
    qpos_goal=qpos,
    num_waypoints=200,  # 200 waypoints × 0.01s = 2 seconds of smooth motion
) # Output: path --> a sequence of 200 joint configurations connecting current pose to goal

# 2.2 Visualize the planned path (for debugging)
path_debug = scene.draw_debug_path(path, franka)

# (3). Step 3: Execute the Path
for waypoint in path:
    franka.control_dofs_position(waypoint) # Send position command from motion planning
    scene.step()    # Simulate 10ms of physics
    camera.render()  # render camera for video recording

# (4). Step 4: Stabilization
# remove the drawn path
scene.clear_debug_object(path_debug)

# allow controller to settle (100 steps = 1 second)
#  Note that after we execute the path, we let the controller run for another 100 steps.
#  This is because we are using a PD controller, and there will be a gap between the desired target position and the current position.
for i in range(100):
    scene.step()
    camera.render()  # render camera for video recording


###########################################################################
#  PHASE 2: REACH (Lower to Object)
###########################################################################

# (1). Step 1: IK for Reach Position
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.130]),  # Lower to 13cm
    quat=np.array([0, 1, 0, 0]), # Still pointing down
)
print(qpos) # Debug: show computed joint angles

# (2). Step 2: Execute Reach Motion 
franka.control_dofs_position(qpos[:-2], motors_dof) # Command only arm joints (not gripper)
for i in range(100):
    scene.step()
    camera.render()  # render camera for video recording


###########################################################################
#  PHASE 3: GRASP (Close Gripper)
###########################################################################

# Hold arm position
franka.control_dofs_position(qpos[:-2], motors_dof)

# Apply closing force to gripper
franka.control_dofs_force( # NOTE: FORCE CONTROL NOT POSITION CONTROL for GRASPING
    np.array([-0.5, -0.5]), fingers_dof
)  #  applied a 0.5N grasping force.

for i in range(100):
    scene.step()
    camera.render()  # render camera for video recording

###########################################################################
#  PHASE 3:  LIFT (Raise Object)
###########################################################################


# (1). Step 1: IK for Lift Position
# Lift position: Hand lifted to 28cm height
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.28]), # Raise to 28cm
    quat=np.array([0, 1, 0, 0]), # Still pointing down
)
print(qpos)

# (2). Step 2: Execute Lift
franka.control_dofs_position(qpos[:-2], motors_dof)
for i in range(200):
    scene.step()
    camera.render()  # render camera for video recording

# ------------------------------------------------------------------------
# THE GRASPING SEQUENCE DONE
# ------------------------------------------------------------------------


# Stop video recording and save
print("Stopping video recording...")
camera.stop_recording(save_to_filename="franka_grasp_demo.mp4", fps=60)


"""
python examples/tutorials/IK_motion_planning_grasp.py

"""
