# NOTE: There are two ways for visualizing the scene:
# 1). using the interactive viewer that runs in a separate thread, and
# 2). by manually adding cameras to the scene and render images using the camera.
#
# This tutorial demonstrates method 2 (camera rendering) for headless environments
# where no display is available (e.g., remote development servers).
import genesis as gs


# 1. Initialize Genesis (no display needed)import genesis as gs
gs.init()

# 2. Create a scene with a more detailed viewer and vis setting
# show_viewer = False means that the viewer will not be created --> use this if you want to render images without a display (headless environments)
# show_viewer = True will create a viewer
scene = gs.Scene(
    show_viewer=False,
    viewer_options=gs.options.ViewerOptions(
        res=(
            1280,
            960,
        ),  # If res is set to None, genesis will automatically create a 4:3 window with the height set to half of your display height.
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=60,  #  The viewer will run as fast as possible if max_FPS is set to None.
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,  # visualize the coordinate frame of `world` at its origin
        world_frame_size=1.0,  # length of the world frame in meter
        show_link_frame=False,  # do not visualize coordinate frames of entity links
        show_cameras=False,  # do not visualize mesh and frustum of the cameras added
        plane_reflection=True,  # turn on plane reflection
        ambient_light=(0.1, 0.1, 0.1),  # ambient light setting
    ),
    renderer=gs.renderers.Rasterizer(),  # using rasterizer for camera rendering
)

# 3. Load objects into the scene
plane = scene.add_entity(
    gs.morphs.Plane(),
)
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)

# 4. Add a camera object to the scene
# Cameras are not connected to the viewer or the display, and returns rendered images only when you need it. Therefore, camera works in headless mode.
cam = scene.add_camera(
    res=(640, 480),
    pos=(3.5, 0.0, 2.5),
    lookat=(0, 0, 0.5),
    fov=30,
    GUI=True,  # If GUI=True, each camera will create an opencv window to dynamically display the rendered image. Note that this is different from the viewer GUI.
)

# 6. Build the scene and start simulating
scene.build()

# 7. Render images using the camera
# Our camera supports rendering rgb image, depth, segmentation mask and surface normals.
# By default, only rgb is rendered
rgb, depth, segmentation, normal = cam.render(
    rgb=True, depth=True, segmentation=True, normal=True
)

# 8. Record videos using camera
# start camera recording. Once this is started, all the rgb images rendered will be recorded internally

cam.start_recording()
import numpy as np

for i in range(120):
    scene.step()
    cam.set_pose(
        pos=(3.0 * np.sin(i / 60), 3.0 * np.cos(i / 60), 2.5),
        lookat=(0, 0, 0.5),
    )
    cam.render()
cam.stop_recording(save_to_filename="video.mp4", fps=60)
