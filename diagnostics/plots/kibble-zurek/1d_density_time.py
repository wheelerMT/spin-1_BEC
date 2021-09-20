import numpy as np
import h5py
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
X = data_file['grid/x']
dx = X[1] - X[0]

num_of_frames = psi_plus.shape[-1]

cvals = np.linspace(0, np.max(abs(psi_0[:, 0])**2), 25, endpoint=True)

# Generate time array
dt = data_file['time/dt'][...]
Nframe = data_file['time/Nframe'][...]
time = dt * Nframe * np.arange(1, num_of_frames + 1)

# * Need to generate 2D stack of the 1D density
spacetime_plus = np.empty((len(X), num_of_frames))
spacetime_0 = np.empty((len(X), num_of_frames))
spacetime_minus = np.empty((len(X), num_of_frames))

for i in range(num_of_frames):
    spacetime_plus[:, i] = abs(psi_plus[:, i]) ** 2
    spacetime_0[:, i] = abs(psi_0[:, i]) ** 2
    spacetime_minus[:, i] = abs(psi_minus[:, i]) ** 2

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

ax[0].pcolormesh(time, X, spacetime_plus, vmin=0, vmax=1, shading='auto')
ax[1].pcolormesh(time, X, spacetime_0, vmin=0, vmax=1, shading='auto')
ax[2].pcolormesh(time, X, spacetime_minus, vmin=0, vmax=1, shading='auto')

plt.show()
