import h5py
import numpy as np
import include.vortex_detection as vd
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams.update({'font.size': 12})

# Load in data:
filename = 'frames/100kf_1024nomag_psi_0=0'  # input('Enter data filename: ')
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
frame = 1  # int(input('Enter the frame number you would like to plot: '))

# Calculate macroscopic phase:
dens = abs(psi_plus[:, :, frame]) ** 2 + abs(psi_minus[:, :, frame]) ** 2
theta = (np.angle(psi_plus[:, :, frame]) + np.angle(psi_minus[:, :, frame]))

# Detect vortices:
psqv, nsqv, phqv_plus, nhqv_plus, phqv_minus, nhqv_minus, num_of_vortices = \
    vd.calculate_vortices(psi_plus[:, :, frame], psi_minus[:, :, frame], x, y)

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 8))
plt.subplots_adjust(hspace=0.1)
for ax in axes:
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    if ax == axes[1]:
        ax.set_xlabel(r'$x/\xi$')
    ax.set_ylabel(r'$y/\xi$', labelpad=-8)

phase_plot_sqv = axes[0].contourf(X, Y, theta, levels=50, cmap='gnuplot')

phase_cbar = plt.colorbar(phase_plot_sqv, ax=axes.ravel().tolist(), aspect=30)
phase_cbar.set_ticks([-np.pi, 0, np.pi])
phase_cbar.set_ticklabels([r'$-\pi$', '0', r'$\pi$'])
phase_cbar.ax.tick_params(labelsize=12)


if len(psqv) != 0:
    axes[0].plot(*zip(*psqv), 'wo', markersize=7, label=r'$\sigma_{SQV} = 1$')  # Plots positive SQVs
if len(nsqv) != 0:
    axes[0].plot(*zip(*nsqv), 'ko', markersize=7, label=r'$\sigma_{SQV} = -1$')  # Plots negative SQVs
if len(phqv_plus) != 0:
    axes[0].plot(*zip(*phqv_plus), 'wX', markersize=7, label=r'$\sigma_1 = 1$')  # Positive HQV in psi_plus
if len(nhqv_plus) != 0:
    axes[0].plot(*zip(*nhqv_plus), 'kX', markersize=7, label=r'$\sigma_1 = -1$')  # Negative HQV in psi_plus
if len(phqv_minus) != 0:
    axes[0].plot(*zip(*phqv_minus), 'w^', markersize=7, label=r'$\sigma_{-1} = 1$')  # Positive HQV in psi_minus
if len(nhqv_minus) != 0:
    axes[0].plot(*zip(*nhqv_minus), 'k^', markersize=7, label=r'$\sigma_{-1} = -1$')  # Negative HQV in psi_minus

# Load in data:
filename = 'frames/100kf_1024nomag_newinit'  # input('Enter data filename: ')
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
frame = -1  # int(input('Enter the frame number you would like to plot: '))

theta = (np.angle(psi_plus[:, :, frame]) + np.angle(psi_minus[:, :, frame])) / 2

# Detect vortices:
psqv, nsqv, phqv_plus, nhqv_plus, phqv_minus, nhqv_minus, num_of_vortices = \
    vd.calculate_vortices(psi_plus[:, :, frame], psi_minus[:, :, frame], x, y)

phase_plot_hqv = axes[1].contourf(X, Y, theta, levels=50, cmap='gnuplot')
r"""
phase_cbar = plt.colorbar(phase_plot_sqv, ax=ax[0])
phase_cbar.set_ticks([-np.pi, 0, np.pi])
phase_cbar.set_ticklabels([r'$-\pi$', '0', r'$\pi$'])
phase_cbar.ax.tick_params(labelsize=20)
"""

if len(psqv) != 0:
    axes[1].plot(*zip(*psqv), 'wo', markersize=7, label=r'$\sigma_{SQV} = 1$')  # Plots positive SQVs
if len(nsqv) != 0:
    axes[1].plot(*zip(*nsqv), 'ko', markersize=7, label=r'$\sigma_{SQV} = -1$')  # Plots negative SQVs
if len(phqv_plus) != 0:
    axes[1].plot(*zip(*phqv_plus), 'wX', markersize=7, label=r'$\sigma_1 = 1$')  # Positive HQV in psi_plus
if len(nhqv_plus) != 0:
    axes[1].plot(*zip(*nhqv_plus), 'kX', markersize=7, label=r'$\sigma_1 = -1$')  # Negative HQV in psi_plus
if len(phqv_minus) != 0:
    axes[1].plot(*zip(*phqv_minus), 'w^', markersize=7, label=r'$\sigma_{-1} = 1$')  # Positive HQV in psi_minus
if len(nhqv_minus) != 0:
    axes[1].plot(*zip(*nhqv_minus), 'k^', markersize=7, label=r'$\sigma_{-1} = -1$')  # Negative HQV in psi_minus



save_fig = 'y'  # input('Do you want to save the figure? (y/n): ')
if save_fig == 'y':
    plot_filename = 'phases'  # input('Enter the filename: ')
    plt.savefig('../../images/unsorted/{}.png'.format(plot_filename), dpi=200)
plt.show()
