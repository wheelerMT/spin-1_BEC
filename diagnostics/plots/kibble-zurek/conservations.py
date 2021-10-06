import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import include.diag as diag
matplotlib.use('TkAgg')

# Load in data:
filename = '1d_kibble-zurek/single_runs/1d_polar-BA-FM_250'  # input('Enter data filename: ')
data_file = h5py.File('../../../data/{}.hdf5'.format(filename), 'r')

psi_plus = data_file['wavefunction/psi_plus']
psi_0 = data_file['wavefunction/psi_0']
psi_minus = data_file['wavefunction/psi_minus']
n = abs(psi_plus[...]) ** 2 + abs(psi_0[...]) ** 2 + abs(psi_minus[...]) ** 2

_, _, fz, _ = diag.calculate_spin(psi_plus, psi_0, psi_minus, n)

# Other variables:
X = data_file['grid/x']
Nx = len(X)
dx = X[1] - X[0]
dkx = 2 * np.pi / (Nx * dx)
Kx = np.fft.fftshift(np.arange(-Nx // 2, Nx // 2) * dkx)

# Generate time array
num_of_frames = psi_plus.shape[-1]
dt = data_file['time/dt'][...]
Nframe = data_file['time/Nframe'][...]
time = dt * Nframe * np.arange(1, num_of_frames + 1)
quench_rate = int(filename[-3:])
q_init = 1.5
q = np.empty(num_of_frames)
for i in range(num_of_frames):
    q[i] = q_init * (1 - time[i] / quench_rate)
q = np.where(q < -q_init, -q_init, q)
p = 0

# Calculate conserved quantities:
N = np.empty(num_of_frames)
mag = np.empty(num_of_frames)
energy_kin = np.empty(num_of_frames)
energy_pot = np.empty(num_of_frames)
energy_int = np.empty(num_of_frames)
for i in range(num_of_frames):
    N[i] = int(dx * np.sum(n[:, i]))
    mag[i] = int(dx * np.sum(fz[:, i])) / Nx
    energy_kin[i], energy_pot[i], energy_int[i] \
        = diag.calculate_energy_1d(psi_plus[:, i], psi_0[:, i], psi_minus[:, i], dx, Kx, 1, p, q[i], 0, 10, -0.5)

fig, ax = plt.subplots(3, sharex=True)
ax[0].plot(time, N, 'k')
ax[0].set_ylabel(r'$N$')

ax[1].plot(time, mag, 'k')
ax[1].set_ylabel(r'$M_z$')

ax[2].plot(time, energy_kin + energy_pot + energy_int, 'k')
ax[2].set_xlabel(r'$t/\tau$')
ax[2].set_ylabel(r'$E[\Psi]$')

plt.tight_layout()
plt.show()
