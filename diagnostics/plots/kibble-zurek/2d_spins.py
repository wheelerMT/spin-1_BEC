import numpy as np
import h5py
import include.diag as diag
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Load in data:
filename = input('Enter data filename: ')
data_file = h5py.File('../../../data/{}.hdf5'.format(filename), 'r')

psi_plus = data_file['wavefunction/psi_plus']
psi_0 = data_file['wavefunction/psi_0']
psi_minus = data_file['wavefunction/psi_minus']

# Other variables:
x, y = data_file['grid/x'], data_file['grid/y']
dx, dy = x[1] - x[0], y[1] - y[0]
X, Y = np.meshgrid(x[:], y[:])

num_of_frames = psi_plus.shape[-1]

saved_times = data_file['saved_times']
list_of_times = []
for i in range(psi_plus.shape[-1]):
    list_of_times.append([i, saved_times[i]])
print(tabulate(list_of_times, headers=["Frame #", "Time"], tablefmt="orgtbl"))

# Calculate spin vectors:
n = abs(psi_plus[...]) ** 2 + abs(psi_0[...]) ** 2 + abs(psi_minus[...]) ** 2
fx, fy, fz, F = diag.calculate_spin(psi_plus, psi_0, psi_minus, n)
F_perp = fx + 1j * fy

frame = int(input('Enter which frame you would like to plot: '))
fig, ax = plt.subplots(1, 3, sharey=True, figsize=(10, 10))
for axis in ax:
    axis.set_aspect('equal')

F_plot = ax[0].contourf(X, Y, F[:, :, frame], np.linspace(0, 1.01, 25, endpoint=True), cmap='jet')
ax[0].set_title(r'$|F|$')
F_perp_plot = ax[1].contourf(X, Y, np.angle(F_perp[:, :, frame]), np.linspace(-np.pi, np.pi, 25, endpoint=True), cmap='jet')
ax[1].set_title(r'Angle$(F_\perp)$')
Fz_plot = ax[2].contourf(X, Y, fz[:, :, frame], np.linspace(-1, 1.01, 25, endpoint=True), cmap='jet')
ax[2].set_title(r'$F_z$')

# Set up cbars
F_cbar = plt.colorbar(F_plot, ax=ax[0], fraction=0.042)
F_perp_cbar = plt.colorbar(F_perp_plot, ax=ax[1], fraction=0.042)
F_perp_cbar.set_ticks([-np.pi, 0, np.pi])
F_perp_cbar.set_ticklabels(['-$\pi$', '0', r'$\pi$'])
Fz_cbar = plt.colorbar(Fz_plot, ax=ax[2], fraction=0.042)
Fz_cbar.set_ticks([-1, 1])
Fz_cbar.set_ticklabels(['-1', '1'])
plt.suptitle(r'$\tau = {}$'.format(saved_times[frame]))

plt.show()
