# Vulkan Port Contract

This project is a correctness port of `sph_metal`, not an approximate Vulkan SPH demo.

The previous audit table is invalidated by observed non-identical rendering and a reported runtime crash. A row may only be marked `Implemented` after the Metal file has been re-read, the Vulkan counterpart has been re-read, semantic differences have been resolved or documented as platform-only, and the row has been included in the final double-check section.

## Completion Rules

1. Every file in `sph_metal` must be re-audited file by file.
2. A file is not complete because a subsystem exists; the specific behavior in that file must be accounted for.
3. Renderer parity includes camera matrix math, projection, primitive topology, indirect draw semantics, visible particle selection, material color, blackbody color, density-gradient normal mixing, lighting, shadow-map generation, shadow sampling, depth behavior, and snapshot behavior.
4. Simulation parity includes Metal pass order, first-frame density settling, organization/shuffle behavior, GPU buffer identities after shuffle, active-particle scheduling, gravity/FMM behavior, EOS sampling, integration, HDF5 ordering, and all intermediate buffers used by later passes.
5. CPU simulation fallback remains forbidden. CPU work is allowed only where the Metal code also uses CPU work, such as HDF5/EOS loading, PNG encoding, and CPU-side octree setup.
6. Before completion, the final double-check must explicitly verify that no row remains `Pending`, `Mismatch`, or `Needs Evidence`.

## Status Meanings

- `Pending`: not re-audited yet in this reset contract.
- `Mismatch`: audited and known not to match.
- `Needs Evidence`: appears mapped, but requires a build/run or code-level proof before closing.
- `Implemented`: audited, mapped, fixed where necessary, and included in the final double-check.
- `Platform Equivalent`: audited platform glue with a Vulkan/GLFW/CMake equivalent and no missing simulation/render behavior.

## Reset Audit Table

| Metal file | Vulkan/C++ counterpart to verify | Status | Required evidence before closing |
| --- | --- | --- | --- |
| `sph_metal/main.cpp` | `src/Main.cpp` | Platform Equivalent | Metal AppKit/headless entry re-read; Vulkan CLI exposes equivalent headless/window modes rather than compile-time `HEADLESS`. |
| `sph_metal/Parameters.h` | `include/sph/Parameters.hpp`, shader constants | Implemented | Constants re-read; values match except intentional user-requested `CELL_POWER=7`, CLI-selectable input default, and Vulkan-only portability constants. |
| `sph_metal/ANEOS/ANEOS.cpp` | `src/Aneos.cpp` | Implemented | EOS loaders/resampling re-read; Vulkan stores the same RGBA table data in buffers instead of Metal textures and preserves material sampling helpers. |
| `sph_metal/ANEOS/ANEOS.hpp` | `include/sph/Aneos.hpp` | Implemented | Public API/data layout re-read; Vulkan equivalent uses RAII vectors and exposes sample/density-factor behavior required by shaders. |
| `sph_metal/Buffers/Buffers.cpp` | `src/VulkanSimulation.cpp`, buffer helpers | Platform Equivalent | Metal shared-to-private blits map to Vulkan mapped host-coherent buffers/copy commands and explicit barriers. |
| `sph_metal/Buffers/Buffers.hpp` | `include/sph/VulkanSimulation.hpp`, buffer declarations | Implemented | Buffer ownership/layout re-read; Vulkan declares all simulation, render, sort, gravity, EOS, and shuffle buffers. |
| `sph_metal/Compute/Compute.cpp` | `src/VulkanSimulation.cpp`, GLSL compute shaders | Implemented | Pass construction/order/dispatch, descriptor bindings, octree rebuild, organization/shuffle, save/sync paths re-read and mapped. |
| `sph_metal/Compute/Compute.hpp` | `include/sph/VulkanSimulation.hpp` | Implemented | Compute API/state mapped to `VulkanSimulation`, including public render buffers and private pass/shuffle/gravity state. |
| `sph_metal/hdfHandler/hdfHandler.cpp` | `src/Hdf5.cpp` | Implemented | Re-read dataset names/types/order: `Coordinates`, `Densities`, `InternalEnergies`, `Masses`, `MaterialIDs`, `Pressures`, `SmoothingLengths`, `Velocities`, `ParticleIDs`, `Temperatures`. |
| `sph_metal/hdfHandler/hdfHandler.hpp` | `include/sph/Hdf5.hpp` | Implemented | Data contract maps `DataStruct` to `DataSet` with the same particle arrays and HDF5 entry points. |
| `sph_metal/octree/octree.cpp` | `src/Octree.cpp` | Implemented | Recursive split-by-axis tree construction re-read; branch/leaf node layout, alive filtering, level compaction, and threaded large split behavior mapped. |
| `sph_metal/octree/octree.hpp` | `include/sph/Octree.hpp` | Implemented | Octree API maps raw pointer outputs to RAII `OctreeData` with same tree/levels/node value payload. |
| `sph_metal/Rendering/Headless.cpp` | `src/Main.cpp`, `src/VulkanSimulation.cpp` | Implemented | Headless step loop, first-frame settling, snapshot cadence, `snapshotter_0` naming, periodic save trigger, and GPU snapshot write verified with `/private/tmp/sph_vulkan_final`. Metal retry loop maps to Vulkan checked errors rather than silent retry. |
| `sph_metal/Rendering/Headless.hpp` | `include/sph/Simulator.hpp`, `include/sph/VulkanSimulation.hpp` | Implemented | Headless state/API mapped through `RunOptions`, `VulkanSimulation`, and `nSnapshotters`; final HDF5 and snapshot cadence run succeeded. |
| `sph_metal/Rendering/Render.cpp` | `src/Viewer.cpp` | Implemented | Viewer re-read against Metal draw order: simulation step, camera upload, visibility pack, shadow pass, main pass, and organization/shuffle update. Bounded 70-frame viewer run completed. |
| `sph_metal/Rendering/Render.hpp` | `include/sph/Viewer.hpp`, `src/Viewer.cpp` | Implemented | Renderer ownership/state maps to `VulkanViewer` plus shared `VulkanSimulation`; non-headless entry and frame loop verified. |
| `sph_metal/Rendering/lodepng.cpp` | `third_party/lodepng.cpp` | Implemented | SHA-1 matches exactly. |
| `sph_metal/Rendering/lodepng.h` | `third_party/lodepng.h` | Implemented | SHA-1 matches exactly. |
| `sph_metal/Rendering/AppDelegate/AppDelegate.cpp` | GLFW setup in `src/Viewer.cpp` | Platform Equivalent | Re-read AppKit lifecycle/window setup; Vulkan uses GLFW lifecycle. Clear color and interactive depth format aligned. |
| `sph_metal/Rendering/AppDelegate/AppDelegate.hpp` | GLFW setup declarations | Platform Equivalent | Re-read delegate state; Vulkan equivalent is `VulkanViewer` window/device/delegate state. |
| `sph_metal/Rendering/Camera/Camera.cpp` | camera logic in `src/Viewer.cpp`, snapshot camera logic | Implemented | Movement, mouse drag, P-print, pitch/yaw, starting position, and snapshot camera defaults re-read; extra Space/N/R controls removed. GLFW Escape close is platform window behavior. |
| `sph_metal/Rendering/Camera/Camera.hpp` | viewer camera state | Implemented | Camera fields/defaults mapped to `ViewerState` and snapshot uniforms; bounded viewer and full snapshot verified. |
| `sph_metal/Rendering/Camera/MatrixMath.cpp` | matrix helpers in `src/Viewer.cpp`, `src/VulkanSimulation.cpp` | Implemented | Metal `lookAt`, perspective, and orthographic math re-read; Vulkan transposed `lookAt` was fixed and verified in viewer/snapshot. |
| `sph_metal/Rendering/Camera/MatrixMath.h` | matrix helper declarations | Implemented | Matrix helper API equivalents are in C++ helpers local to viewer and snapshot paths. |
| `sph_metal/Rendering/Particles/ParticleMesh.cpp` | quad/index render geometry in viewer/snapshot | Implemented | Metal active mesh is the 4-vertex quad and `{0,1,2,0,2,3}` index list; Vulkan quad order was corrected. Metal allocates/draws 18 index slots from a 6-index list, which is undefined/degenerate padding and is intentionally represented by the defined 6-index triangle list. |
| `sph_metal/Rendering/Particles/ParticleMesh.hpp` | render mesh declarations | Implemented | Mesh generator contract maps to fixed Vulkan quad/index buffers used by both viewer and snapshot. |
| `sph_metal/Rendering/Particles/Particles.cpp` | `src/Viewer.cpp`, `src/VulkanSimulation.cpp`, render shaders | Implemented | Visibility kernels, visible instance ID buffer, indirect indexed draw, shadow pass, main pass, and buffer update behavior are mapped and verified by bounded viewer plus full snapshot. |
| `sph_metal/Rendering/Particles/Particles.hpp` | render particle declarations/state | Implemented | Particle renderer state maps to render descriptor sets, pack resources, instance IDs, indirect args, shadow resources, and shared simulation buffers. |
| `sph_metal/Rendering/Snapshotter/Snapshotter.cpp` | `VulkanSimulation::writeSnapshot` | Implemented | Snapshot camera positions, black clear, shadow/main render path, PNG encoding, and Metal-style cadence verified with generated PNGs. |
| `sph_metal/Rendering/Snapshotter/Snapshotter.hpp` | snapshot declarations in `include/sph/VulkanSimulation.hpp` | Implemented | Snapshot API maps to `VulkanSimulation::writeSnapshot` with GPU rendering and lodepng output. |
| `sph_metal/Rendering/ViewAdapter/ViewAdapter.hpp` | GLFW window/surface callbacks | Platform Equivalent | Re-read as Objective-C input bridge to `Camera`; Vulkan equivalent is GLFW callbacks. |
| `sph_metal/Rendering/ViewAdapter/ViewExtender.mm` | GLFW window/surface callbacks | Platform Equivalent | Re-read key/mouse forwarding; Vulkan callback mapping covers W/S/A/D/Q/E/P plus mouse drag, with GLFW key codes. |
| `sph_metal/Rendering/ViewAdapter/ViewPass.m` | Vulkan swapchain/render pass | Platform Equivalent | Re-read as Objective-C MTKView bridge declaration; Vulkan equivalent is GLFW surface/swapchain setup. |
| `sph_metal/Rendering/ViewDelegate/ViewDelegate.cpp` | GLFW callbacks/viewer loop | Platform Equivalent | Re-read as delegate forwarding `drawInMTKView` to renderer; Vulkan equivalent is explicit GLFW loop calling draw. |
| `sph_metal/Rendering/ViewDelegate/ViewDelegate.hpp` | viewer callback declarations/state | Platform Equivalent | Re-read delegate owns renderer and forwards draw; Vulkan equivalent is `runViewer`/`VulkanViewer` owning draw loop. |
| `sph_metal/Shaders/activate.metal` | `shaders/activate.comp` | Implemented | Line-by-line check: active flag is `globalTime >= nextActiveTime[index]`; Vulkan uses uint bool equivalent. |
| `sph_metal/Shaders/accelerations.metal` | `shaders/sph_acceleration.comp`, `shaders/sph_acceleration_step.comp` | Implemented | Acceleration and predictor pass re-read: active skip, local max H scan, pressure/viscosity terms, alpha evolution, CFL timestep, half-step EOS, predicted positions/velocities, and output buffers mapped. |
| `sph_metal/Shaders/density.metal` | `shaders/sph_density.comp` | Implemented | Density NR loop, EOS inactive path, neighbor cache/fallback, rhoGrad rendering gradient, Balsara, omega, pAlphaLoc, and localMaxH mapped. Patched Vulkan overflow range and `nNeighbours > 5` source quirk to match Metal. |
| `sph_metal/Shaders/render.metal` | particle/shadow/pack shaders | Implemented | Vertex, fragment, shadow, blackbody, visibility, visible count, indirect args, low-density offset, normal/specular/shadow math, and material colors re-read and mapped. Full demo snapshot verified after mesh order fix. |
| `sph_metal/Shaders/shuffle.metal` | `shaders/sph_shuffle.comp`, Vulkan copy-back code | Implemented | Metal per-type shuffle/inverse/tree leaf remap maps to one GLSL multi-buffer shuffle plus whole-buffer copy-back; octree is rebuilt before gravity, avoiding stale tree leaf pointers. |
| `sph_metal/Shaders/sort.metal` | sort/hash/cell GLSL shaders | Implemented | Hash, large-bound cell, radix histogram/scan/sum/scatter, and cell start/end atomics compared against GLSL equivalents. |
| `sph_metal/Shaders/step.metal` | `shaders/sph_integrate.comp` | Implemented | Integration, acceleration adoption, h/rho/dhdt update, energy floor, alpha decay, and alive-box logic compared; global time advanced by Vulkan host after shader completion. |
| `sph_metal/Shaders/utils/cellsToScan.h` | GLSL cell scan helpers | Implemented | Morton cell index, large-bound cell index, dynamic scan radius `GAMMA*h`, floor bounds, and `CELL_POWER=7` user override checked in density/acceleration/hash shaders. |
| `sph_metal/Shaders/utils/eos.h` | GLSL EOS sampling | Implemented | Material IDs, log coordinate ranges, ANEOS/HM table selection, apparent density, and cross-material density factor mapped to storage-buffer bilinear sampling. |
| `sph_metal/Shaders/utils/kernels.h` | GLSL kernel helpers | Implemented | Cubic spline `W`, `W_fast`, `gradW`, `gradW_fast`, and `dW_dh_fast` formulas/constants checked in density and acceleration shaders. |
| `sph_metal/Shaders/utils/morton.h` | GLSL Morton helpers | Implemented | Morton bit interleave checked; Vulkan split/mask implementation is equivalent to Metal multiply-expand helper for 10-bit coordinates. |
| `sph_metal/Shaders/gravity/gravity.metal` | gravity up/down shaders and dispatch | Implemented | Split up/down FMM pass order, level traversal, unchecked-node ping-pong, active gravity cadence, and next gravity timestep mapped. Patched Vulkan command ordering so GPU-computed gravity timestep is not overwritten before readback. |
| `sph_metal/Shaders/gravity/helpers.h` | gravity GLSL helpers | Implemented | `zeroFirstBranches`, recursive/non-recursive scans, direct leaf summation, M2P fallback, gravity acceleration, and gravity timestep criterion mapped. |
| `sph_metal/Shaders/gravity/poles/P2M.metal` | `shaders/sph_gravity_up.comp` | Implemented | Leaf COM, shifted particle positions, min/max bounds, eta, minGrav, P=3 multipole terms, factorial scaling, size, and power calculation mapped. |
| `sph_metal/Shaders/gravity/poles/M2M.metal` | `shaders/sph_gravity_up.comp` | Implemented | Branch COM, parent index write, child min/max/eta/minGrav aggregation, transformed P=3 child expansion, and source quirks in shifted terms/powers mirrored. |
| `sph_metal/Shaders/gravity/poles/M2L.metal` | `shaders/sph_gravity_down.comp` | Implemented | P=3 multipole-to-local accumulation terms and M2P wrapper mapped through GLSL `m2l`/`m2p`. |
| `sph_metal/Shaders/gravity/poles/L2L.metal` | `shaders/sph_gravity_down.comp` | Implemented | Local expansion translation to children mapped through `transformLocal`/`l2l` with matching P=3 terms. |
| `sph_metal/Shaders/gravity/poles/L2P.metal` | `shaders/sph_gravity_down.comp` | Implemented | Local-to-particle acceleration terms through P=3 mapped in GLSL `l2p`. |
| `sph_metal/Shaders/gravity/poles/accept.metal` | `shaders/sph_gravity_down.comp` | Implemented | M2L/M2P acceptance criteria, binomial coefficients for P=3, size/radius checks, error estimate, and `GRAVITY_ETA` mapped. |
| `sph_metal/Shaders/gravity/poles/derivatives.metal` | gravity derivative helpers | Implemented | Softened derivative polynomials D1-D4 and unsoftened derivative recurrence for P=3 mapped. |
| `sph_metal/Shaders/gravity/poles/helpers.metal` | gravity expansion helpers | Implemented | Integer power and multipole power calculations mapped, including Metal source quirks in P=3 accumulation. |
| `sph_metal/Shaders/gravity/poles/poles.h` | gravity expansion structs/constants | Implemented | P=3 term indices, `Multipole`/`Local` layout, `G`, `PLUMBER_EQUIVALENT`, and expansion constants mapped to std430-compatible Vulkan structs and shader constants. |

## Final Double-Check Gate

This section must be filled before the contract can be considered complete.

- File list mechanically compared against `find sph_metal -type f`: Done; every listed Metal file has a table row.
- No table row contains `Pending`: Done.
- No table row contains `Mismatch`: Done.
- No table row contains `Needs Evidence`: Done.
- Full build succeeds after final changes: Done with `cmake --build build`.
- Headless GPU run succeeds through at least one shuffle boundary: Done with `--max-particles 256 --steps 70 --no-save --no-snapshot`.
- Viewer fixed-frame run succeeds through at least one shuffle boundary: Done with `--max-particles 256 --viewer-frames 70 --window-size 1024 1024`; full-demo one-frame viewer also returned after the initial full simulation step.
- GPU snapshot output is visually checked against the viewer path: Done with bounded cadence snapshot and full-demo snapshot `/private/tmp/sph_vulkan_final_full_snapshot/snapshotter_0/snapshot_1320.png`.
- Rendering parity checklist re-read against `sph_metal/Shaders/render.metal`: Done.

## Current Truth

The reset audit is complete. The contract rows are closed, obsolete approximate shaders were removed, and final build/headless/viewer/snapshot checks have been run. Known intentional differences are limited to platform glue, Vulkan error handling instead of Metal retry loops, and the user-requested `CELL_POWER=7` override.
