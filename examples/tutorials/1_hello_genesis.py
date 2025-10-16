import genesis as gs


# Precision level: By default, genesis uses f32 precision. You can change to f64 if you want a higher precision level by setting precision='64'.

# 1. Initialization
gs.init(backend=gs.gpu)  # gs.cuda # gs.gpu
# gs.init(
#     seed                = None,
#     precision           = '32',
#     debug               = False,
#     eps                 = 1e-12,
#     logging_level       = None,
#     backend             = gs.gpu,
#     theme               = 'dark',
#     logger_verbose_time = False
# )

# 2. Create a scene
# A scene wraps a simulator object, which handles all the underlying physics solvers
#  and a visualizer object, which manages visualization-related concepts.
scene = gs.Scene()

# scene = gs.Scene(
#     sim_options=gs.options.SimOptions(
#         dt=0.01,
#         gravity=(0, 0, -10.0),
#     ),
#     show_viewer=True,
#     viewer_options=gs.options.ViewerOptions(
#         camera_pos=(3.5, 0.0, 2.5),
#         camera_lookat=(0.0, 0.0, 0.5),
#         camera_fov=40,
#     ),
# )

# 3. Load objects into the scene
# In genesis, all the objects and robots are represented as Entity.
# The first parameter for add_entity is morph.
# A morph in Genesis is a hybrid concept, encapsulating both the geometry and pose information of an entity.
plane = scene.add_entity(
    gs.morphs.Plane(),
)
franka = scene.add_entity(
    # gs.morphs.URDF(
    #     file='urdf/panda_bullet/panda.urdf',
    #     fixed=True,
    # ),
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)


# When loading from external files, you need to specify the file location using the file parameter.
# When parsing this, we support both absolute and relative file path.
# Note that since genesis also comes with an internal asset directory (genesis/assets), so if a relative path is used, we search not only relative path with respect to your current working directory, but also under genesis/assets.
# Therefore, in this example, we will retrieve the franka model from: genesis/assets/xml/franka_emika_panda/panda.xml.
# franka = scene.add_entity(
#     gs.morphs.MJCF(
#         file  = 'xml/franka_emika_panda/panda.xml',
#         pos   = (0, 0, 0),
#         euler = (0, 0, 90), # we follow scipy's extrinsic x-y-z rotation convention, in degrees,
#         # quat  = (1.0, 0.0, 0.0, 0.0), # we use w-x-y-z convention for quaternions,
#         scale = 1.0,
#     ),
# )

# 4. Build the scene and start simulating
# Genesis uses just-in-time (JIT) technology to compile GPU kernels on the fly for each run, so we need an explicit step to initiate this process, which puts everything in place, allocates device memory, and creates underlying data fields for simulation.
scene.build()
for i in range(1000):
    scene.step()
