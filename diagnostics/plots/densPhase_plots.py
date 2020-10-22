import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
matplotlib.use('TkAgg')

"""File that generates plots of the atomic density and the argument of the respective wavefunctions."""

# Load in data:
filename = input('Enter data filename: ')
data_file = h5py.File('../../data/{}.hdf5'.format(filename), 'r')

psi_plus = data_file['wavefunction/psi_plus']
psi_0 = data_file['wavefunction/psi_0']
psi_minus = data_file['wavefunction/psi_minus']

# Other variables:
x, y = data_file['grid/x'], data_file['grid/y']
dx, dy = x[1] - x[0], y[1] - y[0]
X, Y = np.meshgrid(x[:], y[:])

num_of_frames = psi_plus.shape[-1]

cvals = np.linspace(0, np.max(abs(psi_plus[:, :, -1])**2), 25, endpoint=True)

saved_times = data_file['saved_times']
list_of_times = []
for i in range(psi_plus.shape[-1]):
    list_of_times.append([i, saved_times[i]])
print(tabulate(list_of_times, headers=["Frame #", "Time"], tablefmt="orgtbl"))

frame = int(input('Enter which frame you would like to plot: '))
fig, ax = plt.subplots(2, 3, sharey=True, figsize=(20, 10))


ax[0, 0].contourf(X, Y, abs(psi_plus[:, :, frame])**2, cvals, cmap='gnuplot')
ax[0, 0].set_title(r'$|\psi_+|^2$')
ax[0, 1].contourf(X, Y, abs(psi_0[:, :, frame])**2, cvals, cmap='gnuplot')
ax[0, 1].set_title(r'$|\psi_0|^2$')
dens_plot = ax[0, 2].contourf(X, Y, abs(psi_minus[:, :, frame])**2, cvals, cmap='gnuplot')
ax[0, 2].set_title(r'$|\psi_-|^2$')
plt.colorbar(dens_plot, ax=ax[0, 2], fraction=0.045)

ax[1, 0].contourf(X, Y, np.angle(psi_plus[:, :, frame]), levels=25, cmap='gnuplot')
ax[1, 0].set_title(r'$Arg(\psi_+)$')
ax[1, 1].contourf(X, Y, np.angle(psi_0[:, :, frame]), levels=25, cmap='gnuplot')
ax[1, 1].set_title(r'$Arg(\psi_0)$')
phase_plot = ax[1, 2].contourf(X, Y, np.angle(psi_minus[:, :, frame]), levels=25, cmap='gnuplot')
ax[1, 2].set_title(r'$Arg(\psi_-)$')
plt.colorbar(phase_plot, ax=ax[1, 2], fraction=0.045)
plt.suptitle(r'$\tau = {}$'.format(saved_times[frame]))
plt.show()
