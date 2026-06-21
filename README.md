# SPH Vulkan

This is a CMake C++ port of the `sph_metal` project layout. It keeps the same HDF5 input/output datasets, the same parameter values, the same aligned `float3` storage layout used by GPU buffers, and the same EOS table preprocessing.

## Build

```sh
cmake -S . -B build
cmake --build build -j4
```

Dependencies:

- CMake 3.24+
- C++20 compiler
- Vulkan SDK, including `glslc`
- HDF5 C library

The build compiles the GLSL compute shaders in `shaders/` to SPIR-V into the build directory.

Gravity multipole order is controlled by `sph::params::multipoleExpansionPower` in `include/sph/Parameters.hpp`.
The Vulkan port supports orders 1 through 4 and derives `N_EXPANSION_TERMS` from that value, matching the Metal parameter layout.

## Run

```sh
build/sph_vulkan --validate --max-particles 256
build/sph_vulkan
build/sph_vulkan --headless --steps 3 --no-save --no-snapshot
```

Useful flags:

- `--input PATH`: HDF5 initial conditions. Defaults to `demo_impact_n50.hdf5`.
- `--output PATH`: final HDF5 save path.
- `--snapshot-dir PATH`: PNG snapshot output directory.
- `--video-interval-seconds N`: encode headless snapshot videos every N simulation seconds; defaults to `1000`, and `0` disables interim video writes.
- `--steps N`: number of simulation steps.
- `--max-particles N`: cap particle count for quick tests.
- `--window`: open the interactive viewer. This is the default.
- `--headless`: run the Vulkan GPU simulation without opening a window.
- `--window-size W H`: set the viewer size.
- `--viewer-frames N`: render a fixed number of viewer frames and exit; useful for smoke tests.
- `--no-snapshot`: skip PNG output.
- `--no-save`: skip HDF5 output.
- `--require-vulkan`: fail if a Vulkan runtime cannot be created.
- `--info`: print the selected Vulkan device.

Viewer controls:

- Space: pause or run the GPU simulation.
- `N`: single-step the GPU simulation.
- `W`/`S`: forward/back.
- `A`/`D`: strafe.
- `Q`/`E`: up/down.
- Drag: rotate, with the same sensitivity as the Metal camera.
- `P`: print camera position and pitch/yaw.
- `R`: reset camera.
- Esc: close.

## GPU Path

The viewer and headless runner use Vulkan for simulation stepping. The CPU simulation fallback is disabled: `Simulator::step()` and `Simulator::run()` fail if called directly, and the CLI has no `--cpu-sim` path. Headless runs synchronize the final GPU state back to the existing HDF5/snapshot output code. The interactive viewer renders directly from live GPU simulation buffers.
