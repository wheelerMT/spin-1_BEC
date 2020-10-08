import h5py
import numpy as np
import matplotlib.pyplot as plt

filename = input('Enter filename: ')
data_1 = h5py.File('../../data/nomag.hdf5'.format(filename), 'r')
data_2 = h5py.File('../../data/mag10%.hdf5'.format(filename), 'r')
data_3 = h5py.File('../../data/mag15%.hdf5'.format(filename), 'r')

# Load grid data_1:
x, y = data_1['grid/x'], data_1['grid/y']
dx, dy = x[1] - x[0], y[1] - y[0]
Nx, Ny = x.size, y.size

# Load wavefunction data_1:
psi_plus_nomag = data_1['wavefunction/psi_plus']
psi_0_nomag = data_1['wavefunction/psi_0']
psi_minus_nomag = data_1['wavefunction/psi_minus']

psi_plus_mag10 = data_2['wavefunction/psi_plus']
psi_0_mag10 = data_2['wavefunction/psi_0']
psi_minus_mag10 = data_2['wavefunction/psi_minus']

psi_plus_mag15 = data_3['wavefunction/psi_plus']
psi_0_mag15 = data_3['wavefunction/psi_0']
psi_minus_mag15 = data_3['wavefunction/psi_minus']

num_of_frames = psi_plus_nomag.shape[-1] // 2

# Load time variables:
dt = data_1['time/dt'][...]
Nframe = data_1['time/Nframe'][...]

# Calculate atom number:
N_plus_nomag = np.empty(num_of_frames, dtype='int')
N_0_nomag = np.empty(num_of_frames, dtype='int')
N_minus_nomag = np.empty(num_of_frames, dtype='int')

N_plus_mag10 = np.empty(num_of_frames, dtype='int')
N_0_mag10 = np.empty(num_of_frames, dtype='int')
N_minus_mag10 = np.empty(num_of_frames, dtype='int')

N_plus_mag15 = np.empty(num_of_frames, dtype='int')
N_0_mag15 = np.empty(num_of_frames, dtype='int')
N_minus_mag15 = np.empty(num_of_frames, dtype='int')

time = np.empty(num_of_frames, dtype='float32')

for i in range(num_of_frames):
    N_plus_nomag[i] = int(dx * dy * np.sum(abs(psi_plus_nomag[:, :, i])**2))
    N_0_nomag[i] = int(dx * dy * np.sum(abs(psi_0_nomag[:, :, i]) ** 2))
    N_minus_nomag[i] = int(dx * dy * np.sum(abs(psi_minus_nomag[:, :, i]) ** 2))

    N_plus_mag10[i] = int(dx * dy * np.sum(abs(psi_plus_mag10[:, :, i]) ** 2))
    N_0_mag10[i] = int(dx * dy * np.sum(abs(psi_0_mag10[:, :, i]) ** 2))
    N_minus_mag10[i] = int(dx * dy * np.sum(abs(psi_minus_mag10[:, :, i]) ** 2))

    N_plus_mag15[i] = int(dx * dy * np.sum(abs(psi_plus_mag15[:, :, i]) ** 2))
    N_0_mag15[i] = int(dx * dy * np.sum(abs(psi_0_mag15[:, :, i]) ** 2))
    N_minus_mag15[i] = int(dx * dy * np.sum(abs(psi_minus_mag15[:, :, i]) ** 2))

    time[i] = (i + 1) * dt * Nframe
    print('On frame number %i.' % i)

fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharey=True)
for axis in ax:
    axis.set_xlabel(r'$\tau$')
    axis.set_aspect('equal')

ax[0].set_ylabel(r'$N$')
ax[0].plot(time, N_plus_nomag, 'r', label=r'$N_+$')
ax[0].plot(time, N_0_nomag, 'g', label=r'$N_0$')
ax[0].plot(time, N_minus_nomag, 'b', label=r'$N_-$')
ax[0].plot(time, N_plus_nomag + N_0_nomag + N_minus_nomag, 'k', label=r'$N_{tot}$')
ax[0].set_title('No magnetisation.')

ax[1].plot(time, N_plus_mag10, 'r', label=r'$N_+$')
ax[1].plot(time, N_0_mag10, 'g', label=r'$N_0$')
ax[1].plot(time, N_minus_mag10, 'b', label=r'$N_-$')
ax[1].plot(time, N_plus_mag10 + N_0_mag10 + N_minus_mag10, 'k', label=r'$N_{tot}$')
ax[1].set_title('10% magnetisation.')

ax[2].plot(time, N_plus_mag15, 'r', label=r'$N_+$')
ax[2].plot(time, N_0_mag15, 'g', label=r'$N_0$')
ax[2].plot(time, N_minus_mag15, 'b', label=r'$N_-$')
ax[2].plot(time, N_plus_mag15 + N_0_mag15 + N_minus_mag15, 'k', label=r'$N_{tot}$')
ax[2].set_title('15% magnetisation.')

ax[0].legend()
plt.savefig('../images/{}_atomnum.png'.format(filename))
