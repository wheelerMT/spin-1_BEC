import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
matplotlib.use('TkAgg')

# Load in data:
filename = input('Enter data filename: ')
data_file = h5py.File('../../data/{}.hdf5'.format(filename), 'r')
diag_file = h5py.File('../../data/diagnostics/{}_diag.hdf5'.format(filename), 'r')

psi_plus = data_file['wavefunction/psi_plus']
psi_0 = data_file['wavefunction/psi_0']
psi_minus = data_file['wavefunction/psi_minus']

spin_expec_mag = diag_file['spin/spin_expectation']

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
fig, ax = plt.subplots(1, 3, sharey=True, figsize=(20, 10))

for axis in ax:
    axis.set_aspect('equal')

# Density plots:
ax[0].contourf(X, Y, abs(psi_plus[:, :, frame])**2, cvals, cmap='gnuplot')
ax[0].set_title(r'$|\psi_+|^2$')
ax1 = ax[1].contourf(X, Y, abs(psi_minus[:, :, frame])**2, cvals, cmap='gnuplot')
plt.colorbar(ax1, ax=ax[1], fraction=0.045)
ax[1].set_title(r'$|\psi_-|^2$')
ax02 = ax[2].contourf(X, Y, spin_expec_mag[:, :, frame], np.linspace(0, 1, 25, endpoint=True), cmap='PuRd')
ax[2].set_title(r'$|<\vec{F}>|$')
plt.colorbar(ax02, ax=ax[2], fraction=0.045)

plt.suptitle(r'$\tau={}$'.format(saved_times[frame]), y=0.8)
plt.subplots_adjust()
save_image = input('Do you want to save the plot? (y/n): ')
if save_image == 'y':
    image_name = input('Enter the name of plot: ')
    plt.savefig('../../images/unsorted/{}.png'.format(image_name), dpi=200)
plt.show()
