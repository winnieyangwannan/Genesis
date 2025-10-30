# Genesis RayTracer Renderer Setup Guide for Linux

## Table of Contents
1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation Guide](#installation-guide)
4. [Code Examples](#code-examples)
5. [Troubleshooting](#troubleshooting)
6. [Official Documentation Links](#official-documentation-links)

---

## Overview

The Genesis RayTracer renderer provides photorealistic, physically-accurate rendering using ray tracing technology. It supports HDR environment maps for image-based lighting and is built on LuisaRender, a high-performance rendering framework.

**Key Features:**
- Photorealistic ray-traced rendering
- HDR environment map support (`.hdr`, `.exr` files)
- Image-based lighting from environment maps
- Configurable ray tracing quality (depth, samples, etc.)
- GPU-accelerated rendering on NVIDIA GPUs

---

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (RTX series recommended)
- **RAM**: 16GB+ recommended
- **Storage**: ~10GB for build dependencies and compilation

### Software
- **OS**: Linux (Ubuntu 20.04, 22.04, or 24.04 recommended)
- **CUDA**: Version 11.8 or 12.x
- **Python**: 3.9, 3.10, 3.11, or 3.12
- **GCC/G++**: Version 11 or higher
- **CMake**: Version 3.26 or higher
- **NVIDIA Driver**: Latest stable version

---

## Installation Guide

### Step 1: Update System and Install Build Tools

```bash
# Update package lists
sudo apt update && sudo apt upgrade -y

# Install build essentials
sudo apt install -y build-essential manpages-dev software-properties-common

# Add GCC 11 repository and install
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install -y gcc-11 g++-11

# Set GCC 11 as default
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 110
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110

# Verify versions
gcc --version  # Should show 11.x or higher
g++ --version  # Should show 11.x or higher
```

### Step 2: Install CMake

```bash
# Install CMake 3.26+ using snap
sudo snap install cmake --classic

# Verify version
cmake --version  # Should show 3.26 or higher
```

### Step 3: Install System Dependencies

```bash
# Install required libraries for LuisaRender
sudo apt install -y \
    vulkan-tools \
    vulkan-validationlayers-dev \
    libx11-dev \
    libxrandr-dev \
    uuid-dev \
    zlib1g-dev \
    pkg-config
```

### Step 4: Install PyTorch with CUDA

```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# OR for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 5: Clone Genesis Repository

```bash
# Navigate to your desired installation directory
cd ~/

# Clone the Genesis repository
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
cd Genesis
```

### Step 6: Install Genesis with Render Support

```bash
# Install Genesis in editable mode with render dependencies
pip install -e '.[render]'
```

### Step 7: Build LuisaRender

```bash
# Navigate to LuisaRender directory
cd genesis/ext/LuisaRender

# Configure build with CMake (adjust PYTHON_VERSIONS to match your Python version)
cmake -S . -B build \
    -D CMAKE_BUILD_TYPE=Release \
    -D PYTHON_VERSIONS=3.12 \
    -D LUISA_COMPUTE_DOWNLOAD_NVCOMP=ON \
    -D LUISA_COMPUTE_ENABLE_GUI=OFF \
    -D LUISA_RENDER_BUILD_TESTS=OFF

# Build (this takes 10-30 minutes depending on your system)
cmake --build build -j $(nproc)

# Return to Genesis root directory
cd ../../..
```

### Step 8: Verify Installation

```bash
# Test that LuisaRender is properly installed
python -c "import genesis as gs; import LuisaRenderPy; print('✓ LuisaRender successfully installed!')"

# Test basic Genesis functionality
python -c "import genesis as gs; gs.init(); print('✓ Genesis initialized successfully!')"
```

---

## Code Examples

### Example 1: Basic Scene with HDR Environment Map

```python
import genesis as gs

########################## Initialize Genesis ##########################
gs.init(backend=gs.cuda)  # Use CUDA backend

########################## Create Environment Surface ##########################
# Create a surface with your HDR texture
env_surface = gs.surfaces.Default(
    diffuse_texture=gs.textures.ImageTexture(
        image_path="/path/to/your/environment.hdr",  # Path to your HDR file
        encoding='linear',  # Use 'linear' for HDR files, 'srgb' for regular images
    ),
    color=(1.0, 1.0, 1.0, 1.0),  # White multiplier (no tinting)
)

########################## Create Scene with RayTracer ##########################
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,  # Time step: 10ms
        gravity=(0, 0, -10.0),  # Gravity in m/s²
    ),
    renderer=gs.renderers.RayTracer(
        # Environment map configuration
        env_surface=env_surface,  # HDR environment texture
        env_radius=1000.0,  # Radius of environment sphere (meters)
        env_pos=(0.0, 0.0, 0.0),  # Environment position
        env_euler=(0.0, 0.0, 0.0),  # Environment rotation (degrees)
        
        # Optional: Additional point lights
        lights=[
            {
                'pos': (0.0, 0.0, 10.0),
                'color': (1.0, 1.0, 1.0),
                'intensity': 10.0,
                'radius': 4.0,
            }
        ],
        
        # Ray tracing quality settings
        tracing_depth=32,  # Maximum ray bounces (higher = better quality, slower)
        rr_depth=0,  # Russian roulette depth for path termination
        rr_threshold=0.95,  # Russian roulette threshold
    ),
    show_viewer=True,  # Show interactive viewer
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        res=(1280, 960),
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        world_frame_size=1.0,
        plane_reflection=True,
        ambient_light=(0.1, 0.1, 0.1),
    ),
)

########################## Add Entities ##########################
# Add a ground plane
plane = scene.add_entity(
    gs.morphs.Plane(),
)

# Add a robot (Franka Panda arm)
robot = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

########################## Build Scene ##########################
scene.build()

########################## Run Simulation ##########################
for i in range(1000):
    scene.step()

print("Simulation completed!")
```

### Example 2: High-Quality Rendering with Camera

```python
import genesis as gs

########################## Initialize ##########################
gs.init(backend=gs.cuda)

########################## Environment Setup ##########################
env_surface = gs.surfaces.Default(
    diffuse_texture=gs.textures.ImageTexture(
        image_path="/path/to/your/environment.hdr",
        encoding='linear',
    ),
)

########################## Create Scene ##########################
scene = gs.Scene(
    renderer=gs.renderers.RayTracer(
        env_surface=env_surface,
        env_radius=1000.0,
        tracing_depth=64,  # Higher quality for final renders
    ),
    show_viewer=True,
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(4.0, -3.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=35,
    ),
)

########################## Add Entities ##########################
plane = scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

# Add a camera for high-quality offline rendering
cam = scene.add_camera(
    res=(1920, 1080),  # Full HD resolution
    pos=(4.0, -3.0, 2.5),
    lookat=(0, 0, 0.5),
    fov=35,
    GUI=False,  # Headless rendering
)

########################## Build ##########################
scene.build()

########################## Render Video ##########################
cam.start_recording()

for i in range(500):
    scene.step()
    cam.render()  # Render each frame with ray tracing

# Save as video
cam.stop_recording(save_to_filename='output_raytraced.mp4', fps=60)

print("High-quality ray-traced video saved!")
```

### Example 3: Multiple HDR Environments with Rotation

```python
import genesis as gs

def create_scene_with_environment(hdri_path, rotation_euler=(0, 0, 0)):
    """
    Create a Genesis scene with specified HDRI environment.
    
    Args:
        hdri_path: Path to HDR image file (.hdr or .exr)
        rotation_euler: Environment rotation in degrees (x, y, z)
    
    Returns:
        Genesis Scene object
    """
    gs.init(backend=gs.cuda)
    
    # Create environment surface
    env_surface = gs.surfaces.Default(
        diffuse_texture=gs.textures.ImageTexture(
            image_path=hdri_path,
            encoding='linear',
        ),
    )
    
    # Create scene with rotated environment
    scene = gs.Scene(
        renderer=gs.renderers.RayTracer(
            env_surface=env_surface,
            env_radius=1000.0,
            env_euler=rotation_euler,  # Rotate environment
        ),
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
    )
    
    return scene

########################## Usage Examples ##########################

# Example 1: Outdoor sunset with 45° rotation
scene1 = create_scene_with_environment(
    hdri_path="/path/to/sunset_outdoor.hdr",
    rotation_euler=(0, 0, 45)  # Rotate 45° around Z-axis
)

# Example 2: Indoor studio lighting
scene2 = create_scene_with_environment(
    hdri_path="/path/to/studio_lighting.hdr",
    rotation_euler=(0, 0, 0)
)

# Example 3: Forest environment with 90° rotation
scene3 = create_scene_with_environment(
    hdri_path="/path/to/forest_path.hdr",
    rotation_euler=(0, 0, 90)
)

# Add entities and run simulation...
```

### Example 4: Using Regular Images (JPG/PNG) as Environment

```python
import genesis as gs

gs.init(backend=gs.cuda)

# You can also use regular images (not HDR) for environments
env_surface = gs.surfaces.Default(
    diffuse_texture=gs.textures.ImageTexture(
        image_path="/path/to/panorama.jpg",  # JPG or PNG file
        encoding='srgb',  # Use 'srgb' for regular images
    ),
)

scene = gs.Scene(
    renderer=gs.renderers.RayTracer(
        env_surface=env_surface,
        env_radius=1000.0,
    ),
    show_viewer=True,
)

# Note: Regular images won't provide realistic lighting like HDR images
# They're mainly for visual background purposes
```

### Example 5: Adjusting Environment Position and Orientation

```python
import genesis as gs
import numpy as np

gs.init(backend=gs.cuda)

env_surface = gs.surfaces.Default(
    diffuse_texture=gs.textures.ImageTexture(
        image_path="/path/to/environment.hdr",
        encoding='linear',
    ),
)

scene = gs.Scene(
    renderer=gs.renderers.RayTracer(
        env_surface=env_surface,
        env_radius=1000.0,
        env_pos=(0.0, 0.0, 0.0),  # Center position
        
        # Option 1: Use Euler angles (in degrees)
        env_euler=(0.0, 0.0, 45.0),  # Rotate 45° around Z-axis
        
        # Option 2: Use quaternion (w, x, y, z) - commented out
        # env_quat=(0.924, 0.0, 0.0, 0.383),  # Equivalent to 45° Z rotation
    ),
    show_viewer=True,
)

# Add entities and simulate...
```

---

## Troubleshooting

### Issue 1: `ModuleNotFoundError: No module named 'LuisaRenderPy'`

**Cause**: LuisaRender was not built successfully or not found in the correct location.

**Solution**:
```bash
# Verify LuisaRender build exists
ls genesis/ext/LuisaRender/build/bin/

# Rebuild LuisaRender
cd genesis/ext/LuisaRender
rm -rf build  # Clean old build
cmake -S . -B build \
    -D CMAKE_BUILD_TYPE=Release \
    -D PYTHON_VERSIONS=3.12 \
    -D LUISA_COMPUTE_DOWNLOAD_NVCOMP=ON \
    -D LUISA_COMPUTE_ENABLE_GUI=OFF
cmake --build build -j $(nproc)
```

### Issue 2: `GLIBCXX_3.4.30' not found`

**Cause**: Conda's libstdc++ doesn't support the required version.

**Solution**:
```bash
cd $CONDA_PREFIX/lib
mv libstdc++.so.6 libstdc++.so.6.old
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6
```

### Issue 3: Slow Rendering Performance

**Cause**: System may be falling back to CPU rendering instead of GPU.

**Solution**:
```bash
# Force GPU rendering
export LIBGL_ALWAYS_INDIRECT=0
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# Verify GPU is being used
nvidia-smi  # Should show Genesis/Python process using GPU
```

### Issue 4: CMake Cannot Find pybind11

**Cause**: pybind11 is not installed in the environment.

**Solution**:
```bash
pip install "pybind11[global]"
```

### Issue 5: NVTT Warning During Build

**Message**: `NVTT not found! Please install NVTT...`

**Solution**: This warning can usually be ignored. NVTT is optional. If you want to install it:
1. Download from: https://developer.nvidia.com/nvidia-texture-tools-exporter
2. Set environment variable: `export NVTT_DIR=/path/to/nvtt`

### Issue 6: Out of Memory During Compilation

**Cause**: Not enough RAM for parallel compilation.

**Solution**:
```bash
# Reduce number of parallel jobs
cmake --build build -j 4  # Use 4 cores instead of all cores
```

---

## Free HDR Environment Resources

### Recommended HDR Download Sites

1. **Poly Haven** (Highly Recommended)
   - URL: https://polyhaven.com/hdris
   - Quality: Professional, high-resolution HDRIs
   - License: CC0 (Public Domain)
   - Categories: Outdoor, indoor, studio, sunset, forest, urban

2. **HDRI Skies**
   - URL: https://hdri-skies.com/free-hdris/
   - Quality: Sky and cloud HDRIs
   - License: Free for commercial use

3. **HDRMaps Free Section**
   - URL: https://hdrmaps.com/freebies/
   - Quality: Various indoor and outdoor scenes
   - License: Free for commercial and private use

### HDR File Formats

- **`.hdr`**: Standard HDR format (Radiance RGBE)
- **`.exr`**: OpenEXR format (higher quality, larger files)
- **`.jpg`/`.png`**: Regular images (can be used but without true HDR lighting)

---

## Performance Optimization Tips

### 1. Adjust Ray Tracing Quality
```python
# Lower quality for faster preview
renderer=gs.renderers.RayTracer(
    tracing_depth=16,  # Fewer bounces = faster
    env_surface=env_surface,
)

# Higher quality for final renders
renderer=gs.renderers.RayTracer(
    tracing_depth=64,  # More bounces = better quality
    env_surface=env_surface,
)
```

### 2. Use Appropriate Resolution
```python
# Lower resolution for testing
cam = scene.add_camera(res=(640, 480), ...)  # Fast

# Higher resolution for final output
cam = scene.add_camera(res=(1920, 1080), ...)  # Slower
cam = scene.add_camera(res=(3840, 2160), ...)  # 4K, very slow
```

### 3. Enable Kernel Caching
Genesis automatically caches compiled kernels after the first run. Keep the cache directory to speed up subsequent runs.

### 4. Monitor GPU Usage
```bash
# Watch GPU usage in real-time
watch -n 0.5 nvidia-smi
```

---

## Official Documentation Links

### Primary Documentation

1. **Genesis Official Website**
   - URL: https://genesis-embodied-ai.github.io/
   - Description: Project overview and showcase

2. **Genesis Documentation (ReadTheDocs)**
   - URL: https://genesis-world.readthedocs.io/
   - Description: Complete documentation hub

3. **Installation Guide**
   - URL: https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html
   - Description: General installation instructions

4. **Visualization & Rendering Guide** ⭐ **MOST IMPORTANT**
   - URL: https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/visualization.html
   - Description: Detailed LuisaRender build instructions and RayTracer setup

5. **RayTracer API Reference**
   - URL: https://genesis-world.readthedocs.io/en/latest/api_reference/options/renderer/raytracer.html
   - Description: Complete RayTracer parameter documentation

6. **Scene API Reference**
   - URL: https://genesis-world.readthedocs.io/en/latest/api_reference/scene/scene.html
   - Description: Scene creation and configuration

7. **Textures API Reference**
   - URL: https://genesis-world.readthedocs.io/en/latest/api_reference/options/texture/image_texture.html
   - Description: ImageTexture and other texture options

8. **Surfaces API Reference**
   - URL: https://genesis-world.readthedocs.io/en/latest/api_reference/index.html
   - Description: Complete API reference including surfaces

### GitHub Resources

1. **Genesis GitHub Repository**
   - URL: https://github.com/Genesis-Embodied-AI/Genesis
   - Description: Source code and issues

2. **GitHub Issues**
   - URL: https://github.com/Genesis-Embodied-AI/Genesis/issues
   - Description: Bug reports and feature requests (search for "LuisaRender" or "RayTracer")

3. **GitHub Discussions**
   - URL: https://github.com/Genesis-Embodied-AI/Genesis/discussions
   - Description: Community discussions and Q&A

4. **Example Scripts**
   - URL: https://github.com/Genesis-Embodied-AI/Genesis/tree/main/examples
   - Description: Official example scripts

### Related Projects

1. **LuisaRender (Alif-01 Fork)**
   - URL: https://github.com/Alif-01/LuisaRender
   - Description: The specific LuisaRender fork used by Genesis

2. **LuisaCompute**
   - URL: https://github.com/LuisaGroup/LuisaCompute
   - Description: Domain-specific language for high-performance rendering

---

## Quick Reference: Key Parameters

### RayTracer Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `env_surface` | `Surface` | `None` | Surface with texture for environment map |
| `env_radius` | `float` | `1000.0` | Radius of environment sphere (meters) |
| `env_pos` | `tuple` | `(0, 0, 0)` | Position of environment sphere |
| `env_euler` | `tuple` | `(0, 0, 0)` | Euler rotation angles (degrees, x-y-z) |
| `env_quat` | `tuple` | `None` | Quaternion rotation (w, x, y, z) |
| `tracing_depth` | `int` | `32` | Maximum ray bounces (quality) |
| `rr_depth` | `int` | `0` | Russian roulette depth |
| `rr_threshold` | `float` | `0.95` | Russian roulette threshold |
| `lights` | `list` | See docs | Additional point lights |

### ImageTexture Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_path` | `str` | `None` | Path to image file |
| `image_array` | `array` | `None` | Numpy array of image data |
| `image_color` | `tuple` | `(1,1,1,1)` | Color multiplier (RGBA) |
| `encoding` | `str` | `'srgb'` | `'srgb'` or `'linear'` (use linear for HDR) |

---

## Summary Checklist

- [ ] Ubuntu Linux with NVIDIA GPU
- [ ] CUDA 11.8+ or 12.x installed
- [ ] Python 3.9-3.12 installed
- [ ] GCC/G++ 11+ installed
- [ ] CMake 3.26+ installed
- [ ] System dependencies installed (Vulkan, X11, etc.)
- [ ] PyTorch with CUDA installed
- [ ] Genesis cloned from GitHub
- [ ] Genesis installed with `pip install -e '.[render]'`
- [ ] LuisaRender built successfully
- [ ] Installation verified with import test
- [ ] HDR environment files downloaded
- [ ] Test script runs successfully

---

## Support and Community

If you encounter issues not covered in this guide:

1. **Check GitHub Issues**: Search existing issues for similar problems
2. **Read Official Docs**: Review the visualization guide thoroughly
3. **Ask in Discussions**: Post in GitHub Discussions for community help
4. **Report Bugs**: Create a new issue with detailed error messages and system info

---

**Document Version**: 1.0  
**Last Updated**: October 2025  
**Genesis Version**: 0.3.4+

**Note**: This guide is based on Genesis version 0.3.4. Check the official documentation for updates and changes in newer versions.
