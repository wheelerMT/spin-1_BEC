import numpy as np
import h5py
import matplotlib.pyplot as plt

# Load in data:
filename = '1d_polar-BA-FM_5000'  # input('Enter data filename: ')
# data_file = h5py.File('../../../scratch/data/spin-1/kibble-zurek/{}.hdf5'.format(filename), 'r')
data_file = h5py.File('../../../data/1d_kibble-zurek/{}.hdf5'.format(filename), 'r')

psi_plus = data_file['wavefunction/psi_plus']
psi_0 = data_file['wavefunction/psi_0']
psi_minus = data_file['wavefunction/psi_minus']
n_0 = 1.

# Other variables:
X = data_file['grid/x']
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

print('Setting up figure environment...')
fig, ax = plt.subplots(3, 1, sharex=True)
for axis in ax:
    axis.set_ylabel(r'$x/\xi_s$')
    if axis == ax[0]:
        axis.set_title(r'$|\psi_+|^2$')
    if axis == ax[1]:
        axis.set_title(r'$|\psi_0|^2$')
    if axis == ax[2]:
        axis.set_title(r'$|\psi_-|^2$')
        axis.set_xlabel(r'$t/\tau$')

print('Plotting results...')
ax[0].pcolormesh(time, X, spacetime_plus, vmin=0, vmax=n_0, shading='auto')
ax[1].pcolormesh(time, X, spacetime_0, vmin=0, vmax=n_0, shading='auto')
ax[2].pcolormesh(time, X, spacetime_minus, vmin=0, vmax=n_0, shading='auto')
print('Results plotted!')

print('Saving figure...')
plt.savefig('../../../../plots/spin-1/{}_dens.png'.format(filename), bbox_inches='tight')
print('Image {}_dens.png successfully saved!'.format(filename))
plt.show()
