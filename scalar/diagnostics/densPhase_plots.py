import h5py
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import matplotlib
matplotlib.use('TkAgg')


# Load in required data:
filename = input('Enter data filename: ')
data_file = h5py.File('../../data/{}.hdf5'.format(filename), 'r')

x, y = data_file['grid/x'], data_file['grid/y']
Nx, Ny = len(x), len(y)
dx, dy = x[1] - x[0], y[1] - y[0]
X, Y = np.meshgrid(x[:], y[:])

psi = data_file['wavefunction/psi']

# Get info about the saved times of data and prints to screen:
saved_times = data_file['saved_times']
list_of_times = []
for i in range(psi.shape[-1]):
    list_of_times.append([i, saved_times[i]])
print(tabulate(list_of_times, headers=["Frame #", "Time"], tablefmt="orgtbl"))

frame = int(input('Enter the frame number you wish to plot: '))

# Plots:
fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)
for axis in ax:
    axis.set_aspect('equal')

dens_plot = ax[0].contourf(X, Y, abs(psi[:, :, frame])**2, levels=50, cmap='gnuplot')
plt.colorbar(dens_plot, ax=ax[0], fraction=0.046)
ax[0].set_title(r'$|\psi|^2$')
ax[0].set_ylabel(r'$y/\xi$')
ax[0].set_xlabel(r'$x/\xi$')

phase_plot = ax[1].contourf(X, Y, np.angle(psi[:, :, frame]), levels=50, cmap='gnuplot')
phase_cbar = plt.colorbar(phase_plot, ax=ax[1], fraction=0.046)
phase_cbar.set_ticks([-np.pi, 0, np.pi])
phase_cbar.set_ticklabels([r'$-\pi$', '0', r'$\pi$'])
ax[1].set_title(r'Arg($\psi$)')
ax[1].set_xlabel(r'$x/\xi$')

plt.suptitle(r'$\tau={}$'.format(saved_times[frame]), y=0.85)
plt.tight_layout()

save_image = input('Do you wish to save the plot? (y/n): ')
if save_image == 'y':
    image_name = input('Enter name for plot: ')
    plt.savefig('../../images/unsorted/{}.png'.format(image_name), dpi=400)

plt.show()
