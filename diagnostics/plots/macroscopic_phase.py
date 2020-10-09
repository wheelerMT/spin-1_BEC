import h5py
import numpy as np
import include.vortex_detection as vd
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

"""File that constructs the macroscopic phase (EPP phase only) and plots it for a given time of the dataset. Also 
overlays the positions of the vortices in the system over the plot."""

# Load in data:
filename = 'frames/100kf_1024nomag'  # input('Enter data filename: ')
data_file = h5py.File('../../data/{}.hdf5'.format(filename), 'r')

psi_plus = data_file['wavefunction/psi_plus']
psi_minus = data_file['wavefunction/psi_minus']

# Other variables:
x, y = data_file['grid/x'][...], data_file['grid/y'][...]
dx, dy = x[1] - x[0], y[1] - y[0]
X, Y = np.meshgrid(x[:], y[:])

saved_times = data_file['saved_times']
list_of_times = []
for i in range(psi_plus.shape[-1]):
    list_of_times.append([i, saved_times[i]])
print(tabulate(list_of_times, headers=["Frame #", "Time"], tablefmt="orgtbl"))
frame = int(input('Enter the frame number you would like to plot: '))

# Calculate macroscopic phase:
dens = abs(psi_plus[:, :, frame]) ** 2 + abs(psi_minus[:, :, frame]) ** 2
theta = (np.angle(psi_plus[:, :, frame]) + np.angle(psi_minus[:, :, frame]) - np.pi) / 2

# Detect vortices:
psqv, nsqv, phqv_plus, nhqv_plus, phqv_minus, nhqv_minus, num_of_vortices = \
    vd.calculate_vortices(psi_plus[:, :, frame], psi_minus[:, :, frame], x, y)

fig, ax = plt.subplots(1, figsize=(15, 10))
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())
ax.set_xlabel(r'$x/\xi$')
ax.set_ylabel(r'$y/\xi$')
ax.set_title(r'$\tau = {}$'.format(saved_times[frame]))
phase_plot = ax.contourf(X, Y, theta, levels=25, cmap='jet')
phase_cbar = plt.colorbar(phase_plot, ax=ax)
phase_cbar.set_ticks([-np.pi, 0, np.pi])
phase_cbar.set_ticklabels([r'$-\pi$', '0', r'$\pi$'])
phase_cbar.ax.tick_params(labelsize=20)

if len(psqv) != 0:
    ax.plot(*zip(*psqv), 'wo', markersize=10, label=r'$\sigma_{SQV} = 1$')  # Plots positive SQVs
if len(nsqv) != 0:
    ax.plot(*zip(*nsqv), 'ko', markersize=10, label=r'$\sigma_{SQV} = -1$')  # Plots negative SQVs
if len(phqv_plus) != 0:
    ax.plot(*zip(*phqv_plus), 'wX', markersize=10, label=r'$\sigma_1 = 1$')  # Positive HQV in psi_plus
if len(nhqv_plus) != 0:
    ax.plot(*zip(*nhqv_plus), 'kX', markersize=10, label=r'$\sigma_1 = -1$')  # Negative HQV in psi_plus
if len(phqv_minus) != 0:
    ax.plot(*zip(*phqv_minus), 'w^', markersize=10, label=r'$\sigma_{-1} = 1$')  # Positive HQV in psi_minus
if len(nhqv_minus) != 0:
    ax.plot(*zip(*nhqv_minus), 'k^', markersize=10, label=r'$\sigma_{-1} = -1$')  # Negative HQV in psi_minus
ax.legend()
save_fig = input('Do you want to save the figure? (y/n): ')
if save_fig == 'y':
    plot_filename = input('Enter the filename: ')
    plt.savefig('../../images/unsorted/{}.png'.format(plot_filename), dpi=200)
plt.show()
