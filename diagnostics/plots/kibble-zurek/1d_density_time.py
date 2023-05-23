import numpy as np
import h5py
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True
plt.rc("text.latex", preamble=r"\usepackage{fourier}")
plt.rcParams.update({"font.size": 18})

# Load in data:
filename = '1d_BA-FM_1000'  # input('Enter data filename: ')
# data_file = h5py.File('../../../scratch/data/spin-1/kibble-zurek/{}.hdf5'.format(filename), 'r')
data_file = h5py.File('../../../data/{}.hdf5'.format(filename), 'r')

psi_plus = data_file['wavefunction/psi_plus']
psi_0 = data_file['wavefunction/psi_0']
psi_minus = data_file['wavefunction/psi_minus']
n_0 = 1.

# Other variables:
X = data_file['grid/x'][...]
dx = X[1] - X[0]

num_of_frames = psi_plus.shape[-1]

# Generate time array
dt = data_file['time/dt'][...]
N_steps = data_file['time/N_steps'][...]
time = data_file['time/t'][:, 0]

# if "swislocki" in filename:
#     time = dt * N_steps * np.arange(num_of_frames)
# else:
#     time = dt * N_steps * np.arange(-num_of_frames // 2, num_of_frames // 2 + 1)

# * Need to generate 2D stack of the 1D density
spacetime_plus = np.empty((len(X), num_of_frames))
spacetime_0 = np.empty((len(X), num_of_frames))
spacetime_minus = np.empty((len(X), num_of_frames))

for i in range(num_of_frames):
    # print('Calculating density: {} out of {}'.format(i + 1, num_of_frames))
    spacetime_plus[:, i] = abs(psi_plus[:, i]) ** 2
    spacetime_0[:, i] = abs(psi_0[:, i]) ** 2
    spacetime_minus[:, i] = abs(psi_minus[:, i]) ** 2

fig, ax = plt.subplots(3, 1, figsize=(6.4, 3.2))
for axis in ax:
    axis.set_ylim(-64, 64)
    axis.set_aspect('equal')
    axis.tick_params(axis='y', labelsize=16)
    if axis == ax[0]:
        axis.set_xticks([])
    if axis == ax[1]:
        axis.set_ylabel(r'$x/\xi_s$', labelpad=-10)
        axis.set_xticks([])
    if axis == ax[2]:
        axis.set_xlabel(r'$t/\tau$')
        axis.set_xticks([-990, 0, 1000, 2000])
        axis.set_xticklabels(['$-1000$', '$0$', '$1000$', '$2000$'])

extent = time.min(), time.max(), X.min(), X.max()
ax[0].imshow(spacetime_plus, extent=extent, vmin=0, vmax=n_0, aspect='auto',
             interpolation='gaussian')
ax[1].imshow(spacetime_0, extent=extent, vmin=0, vmax=n_0, aspect='auto',
             interpolation='gaussian')
plot = ax[2].imshow(spacetime_minus, extent=extent, vmin=0, vmax=n_0, aspect='auto',
                    interpolation='gaussian')

cbar = fig.colorbar(plot, ax=ax.ravel().tolist(), location='right', anchor=(1.3, 0))
cbar.set_ticks([0, 1])
cbar.set_label(r'$|\psi_m|^2$', labelpad=-10)
plt.subplots_adjust(hspace=0.06)
plt.savefig('../../../../plots/spin-1/BA-FM_densities.pdf', bbox_inches='tight')
plt.show()
