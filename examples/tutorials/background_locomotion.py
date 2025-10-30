"""
python examples/tutorials/background_locomotion.py
"""


import genesis as gs

gs.init()



scene = gs.Scene(
    renderer=gs.renderers.Rasterizer(),
    show_viewer=True,
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
    ),
)


scene.build()


# TODO: set_environment_map may not be available in current Genesis version
# If you need to set an HDR environment, check the latest documentation
scene.viewer.set_environment_map('/Users/winnieyangwn/Documents/Winnie/Genesis/examples/tutorials/rooitou_park_4k.hdr')
# Genesis may have built-in environment maps
for i in range(1000):
    scene.step()