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


########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        max_FPS=60,
    ),
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    show_viewer=False,  # `show_viewer=False` for headless mode
    rigid_options=gs.options.RigidOptions(
        enable_collision=True,
    ),
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)
cube = scene.add_entity(
    gs.morphs.Box(
        size=(0.04, 0.04, 0.04),
        pos=(0.65, 0.0, 0.02),
    )
)
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)

# For Franka robot: 9 values for joint position: 9 degrees of freedom  (7 arm joints + 2 gripper fingers)
motors_dof = np.arange(7) # DOFs 0-6 (arm joints) - 7 joints
fingers_dof = np.arange(7, 9) # # DOFs 7-8 (gripper fingers) - 2 fingers

# set control gains
# Note: the following values are tuned for achieving best behavior with Franka
# Typically, each new robot would have a different set of parameters.
# Sometimes high-quality URDF or XML file would also provide this and will be parsed.

franka.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
) # First 7 values: Arm joint gains (higher values); Last 2 values: Gripper finger gains (lower values)
franka.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
)
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
)
# Add a camera for video recording
camera = scene.add_camera(
    res=(1280, 720),
    pos=(3, -1, 1.5),
    lookat=(0.0, 0.0, 0.5),
    fov=30,
    GUI=False,
)
########################## build ##########################
scene.build()




# Start video recording
print("Starting video recording...")
camera.start_recording()

# get the end-effector link
end_effector = franka.get_link("hand")

# move to pre-grasp pose
# Use inverse kinetics (IK) to solve the joint position given a target end-effector pose
# An array of joint positions (angles) for all degrees of freedom
# For Franka robot: 9 values (7 arm joints + 2 gripper fingers)
# Pre-grasp position: Hand 25cm above the cube
qpos = franka.inverse_kinematics(
    link=end_effector,  # Which link to position (the "hand")
    pos=np.array([0.65, 0.0, 0.25]), # Target 3D position (x, y, z) in world coordinates (in meters)
    quat=np.array([0, 1, 0, 0]), #  Target orientation (quaternion) --> defines which way the gripper points  # Gripper pointing downward
)

# gripper open pos
qpos[-2:] = 0.04

# Motion planning: plan a smooth path to the target joint position (qpos)
# plan path 
# Output: path --> a sequence of 200 joint configurations connecting current pose to goal
path = franka.plan_path(
    qpos_goal=qpos,
    num_waypoints=200,  # 200 waypoints × 0.01s = 2 seconds of smooth motion
)

# draw the planned path
# This draws the planned path in the viewer so you can visually see:
path_debug = scene.draw_debug_path(path, franka)

# execute the planned path
for waypoint in path:
    franka.control_dofs_position(waypoint)
    scene.step()
    camera.render()  # render camera for video recording

# remove the drawn path
scene.clear_debug_object(path_debug)

# allow robot to reach the last waypoint
#  Note that after we execute the path, we let the controller run for another 100 steps.
#  This is because we are using a PD controller, and there will be a gap between the desired target position and the current position.
for i in range(100):
    scene.step()
    camera.render()  # render camera for video recording

# reach
# Reach position: Hand just above the cube at 13cm height
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.130]),
    quat=np.array([0, 1, 0, 0]),
)
print(qpos)
franka.control_dofs_position(qpos[:-2], motors_dof)
for i in range(100):
    scene.step()
    camera.render()  # render camera for video recording

# grasp
franka.control_dofs_position(qpos[:-2], motors_dof)
franka.control_dofs_force(
    np.array([-0.5, -0.5]), fingers_dof
)  #  applied a 0.5N grasping force.

for i in range(100):
    scene.step()
    camera.render()  # render camera for video recording

# lift
# Lift position: Hand lifted to 28cm height
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.28]),
    quat=np.array([0, 1, 0, 0]),
)
print(qpos)
franka.control_dofs_position(qpos[:-2], motors_dof)
for i in range(200):
    scene.step()
    camera.render()  # render camera for video recording

# Stop video recording and save
print("Stopping video recording...")
camera.stop_recording(save_to_filename="franka_grasp_demo.mp4", fps=60)


"""
python examples/tutorials/IK_motion_planning_grasp.py

"""
