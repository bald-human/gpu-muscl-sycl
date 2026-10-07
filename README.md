# GPU-accelerated MUSCL fluid solver (Python + C++/SYCL)

Master's thesis project, Niels Bohr Institute, University of Copenhagen (2025–2026)
**Accelerating Astrophysical Fluid Dynamics with GPUs: MUSCL Solver Acceleration with Python and SYCL**
Supervisor: Troels Haugbølle

An existing Python/NumPy solver for the 3D compressible Euler equations was ported to C++ and then to **SYCL GPU kernels**. The kernels are exposed to Python through **pybind11**, so a simulation is still set up, run and analysed in Python while all the compute runs on the GPU.

<img width="745" height="443" alt="runtime_results" src="https://github.com/user-attachments/assets/d03425d5-630b-47d4-b73e-c105c6f70eb9" />

## Results

| Version | 32³ | 64³ | 128³ | 256³ |
|---|---|---|---|---|
| Pure Python (NumPy)\* | 20.5 s | 140.1 s | 1264.1 s | n/a |
| C++ (direct translation)\* | 24.4 s | 454.0 s | 4241.0 s | n/a |
| SYCL v0 (first GPU version) | 31.2 s | 50.5 s | 90.4 s | 434.4 s |
| SYCL v2 | 2.29 s | 3.61 s | 17.95 s | 139.95 s |
| **SYCL v4 (final)** | **2.14 s** | **3.53 s** | **17.90 s** | **137.80 s** |

\* Python and C++ were run for 1,000 time steps; all SYCL versions for 10,000 time steps.

- **924× speedup** over the original Python implementation at 128³, the largest grid the Python version could complete. The gap widens with problem size.
- About **1.2 billion cell updates per second** at 256³.
- Kernel launches per time step reduced from **30 to 12**, and large device arrays from **19 to 8**.
- HLL Riemann-solver kernel reduced from **1202 ns to 183 ns** (−84%) by kernel fusion.

<img width="749" height="446" alt="Throughput_results" src="https://github.com/user-attachments/assets/200bc070-47cf-4ab0-b5fa-bd5b754e28ce" />

## Numerical method

- Finite-volume **MUSCL-Hancock** scheme (second order) for the 3D Euler equations
- **Monotonized-central (MonCen)** slope limiter
- **HLL** approximate Riemann solver, with one 1D solver reused for all three directions by reordering the velocity components
- **CFL** time step, computed on the GPU with SYCL reductions
- Periodic boundaries

## Implementation highlights

- **Unified Shared Memory**: all device arrays are allocated once at start-up, so there is no allocation inside the time loop
- `SyclQueueManager` class that owns the queue, the simulation parameters and the device memory, exposed to Python via pybind11
- **Kernel fusion**: the predictor and face-reconstruction steps run in one kernel per direction, and the HLL flux in one kernel
- Profiled with **NVIDIA Nsight Compute** (speed-of-light, memory workload, occupancy, **roofline analysis**). The dominant kernels are memory-bound.
- Trade-offs were tested empirically: recomputation vs. memory traffic, register pressure vs. occupancy, and kernel splitting vs. launch overhead. A version that moved the whole time loop into C++ was dropped after profiling showed the Python–C++ call overhead was negligible.

## Repository layout

```
full_muscl_solver.cpp   # final C++/SYCL implementation + pybind11 module `muscl_step`
mesh.py                 # Mesh class: grid, coordinates and state arrays (from the reference Python code)
All-steps.py            # driver: 3D blast-wave initial condition and time loop
CMakeLists.txt          # builds the Python extension with icpx -fsycl
run_sim.sh              # Slurm job script: configure, build, run
original_c++/           # intermediate C++ translation (CPU only)
```

## Build and run

Requirements:
- Intel oneAPI DPC++ compiler (`icpx`)
- CMake ≥ 3.18
- Python 3 with NumPy and pybind11 (`pip install numpy pybind11`)
- a SYCL-capable GPU (developed on NVIDIA GPUs). Without a GPU, the code falls back to a CPU queue.

```bash
mkdir -p build && cd build
cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx ..   # NVIDIA: add -DSYCL_TARGETS=nvptx64-nvidia-cuda
cmake --build . --config Release -j
cd .. && PYTHONPATH=$PWD/build:$PYTHONPATH python All-steps.py
```

On a Slurm cluster: `sbatch run_sim.sh` (adjust the partition name to your cluster).

The default run is a 32³ blast wave (γ = 1.2) that prints the final state at the domain centre. Change `n` in `All-steps.py` for larger grids.

## Future work

- In-order queue to remove host synchronisation after every kernel
- Local (shared) memory for stencil reuse
- Multi-GPU domain decomposition with MPI

## Thesis

The full thesis is available on request.
