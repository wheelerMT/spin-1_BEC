import numpy as np
import h5py
import include.diag as diag
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Load in data:
filename = '1d_kibble-zurek/single_runs/1d_polar-BA-FM_500'  # input('Enter data filename: ')
data_file = h5py.File('../../../data/{}.hdf5'.format(filename), 'r')

# Spatial and time data
x = data_file['grid/x']
Nx = len(x)
dx = x[1] - x[0]
dt = data_file['time/dt'][...]
Nframe = data_file['time/Nframe'][...]
time = dt * Nframe * np.arange(1, 200 + 1)

psi_plus = data_file['wavefunction/psi_plus'][...]
psi_0 = data_file['wavefunction/psi_0'][...]
psi_minus = data_file['wavefunction/psi_minus'][...]
n = abs(psi_plus) ** 2 + abs(psi_0) ** 2 + abs(psi_minus) ** 2

num_of_frames = psi_plus.shape[-1]

# Set up q
quench_rate = int(filename[-3:])
q_init = 1.5
q = np.empty(num_of_frames)
for i in range(num_of_frames):
    q[i] = q_init * (1 - time[i] / quench_rate)
q = np.where(q < -q_init, -q_init, q)
frame_q0 = int(quench_rate / (Nframe * dt))

# Calculate spin vectors:
fx, fy, fz, F = diag.calculate_spin(psi_plus, psi_0, psi_minus, n)
F_plus = fx + 1j * fy
F_minus = fx - 1j * fy

trans_mag = np.empty(num_of_frames)
for i in range(num_of_frames):
    trans_mag[i] = dx * np.sum(abs(fx[:, i]) ** 2 + abs(fy[:, i]) ** 2) / (Nx * dx)

plt.plot(q[:frame_q0], trans_mag[:frame_q0], 'k')
plt.xlabel(r'$q(t)$')
plt.ylabel(r'$M_TL$')
plt.tight_layout()
plt.show()
