import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from time import time
import os

from mesh        import Mesh
import muscl_step

# allow 40 figures to be opened
mpl.rcParams['figure.max_open_warning'] = 40

# stop the code if there are floating point errors
np.seterr(over='raise',invalid='raise',under='ignore')

# simulation parameters
n = [32,32,32]  # resolution of the mesh
center = [0,0,0] # center of the mesh
size = [2,2,2]   # physical size of the mesh
cs = 1
gamma = 1.2
Cdt = 0.2

# create the mesh
m = Mesh(n=n, center=center, size=size, gamma=gamma, cs=cs)

# initial conditions
d0 = 1      # density
u0 = [0,0,0] # velocity
p0 = 1      # pressure

m.D[...] = d0                # density
for idim in range(m.ndim):
    m.M[idim] = d0*u0[idim]  # momentum

if not m.isothermal:         # total energy
    m.E[...] = p0 / (m.gamma - 1) + 0.5*d0*(u0[0]**2 + u0[1]**2 + u0[2]**2)

# adding a blast wave
power = 2
w = 3
e0 = 1e3
d0_blast = 2
blast = np.exp(-np.abs(m.r/(w*m.ds[0]))**power)
blast_int = np.sum(blast * m.ds.prod()) # unormalised total energy of the blast wave
m.E[...] = m.E[...] + e0 * blast / blast_int
m.D[...] = m.D[...] + d0_blast * blast / blast_int

# simulation control parameters
dtmax = 1e-2
tend = 2.0
max_step = 3000

# SYCL queue manager setup
q_mgr = muscl_step.SyclQueuemanager()
q_mgr.init_params(m.n, m.nv, m.ndim, m.isothermal, m.iM, m.cs, m.gamma, Cdt)
q_mgr.init_vars_from_host(m.vars)   # called once to allocate and copy
q_mgr.malloc_arrays(m.ds)

# Run simulation silently
while m.t < tend and m.step < max_step:
    # Compute Courant timestep
    dt_c = muscl_step.Courant(q_mgr)
    dt = min(dt_c, dtmax)
    if m.t + dt > tend:
        dt = tend - m.t
    
    # MUSCL step
    muscl_step.Calc_Step(q_mgr, dt)
    
    # Update simulation state
    m.t += dt
    m.step += 1

# Final data transfer and cleanup
q_mgr.cpy_vars_from_device()
q_mgr.free_arrays()

print(f"\nSimulation complete:")
print(f"Total steps:     {m.step}")
print(f"Final time:      {m.t:.4f}")

# Optional: save results
# output_dir = "./simulation_output"
# os.makedirs(output_dir, exist_ok=True)
# np.save(os.path.join(output_dir, f"vars_final_n{np.prod(n)}_t{m.t:.3f}.npy"), m.vars)
# print(f"Results saved to {output_dir}")
