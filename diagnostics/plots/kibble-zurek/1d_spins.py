import numpy as np
import h5py
import include.diag as diag
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Load in data:
filename = input('Enter data filename: ')
data_file = h5py.File('../../../data/{}.hdf5'.format(filename), 'r')

psi_plus = data_file['wavefunction/psi_plus']
psi_0 = data_file['wavefunction/psi_0']
psi_minus = data_file['wavefunction/psi_minus']
n = abs(psi_plus[...]) ** 2 + abs(psi_0[...]) ** 2 + abs(psi_minus[...]) ** 2

# Other variables:
X = data_file['grid/x']
dx = X[1] - X[0]

num_of_frames = psi_plus.shape[-1]

# Generate time array
dt = data_file['time/dt'][...]
Nframe = data_file['time/Nframe'][...]
time = dt * Nframe * np.arange(1, num_of_frames + 1)

# Calculate spin vectors:
fx, fy, fz, F = diag.calculate_spin(psi_plus, psi_0, psi_minus, n)
F_perp = fx + 1j * fy

# * Need to generate 2D stack of the 1D quantity
spacetime_F = np.empty((len(X), num_of_frames))
spacetime_F_perp = np.empty((len(X), num_of_frames))
spacetime_Fz = np.empty((len(X), num_of_frames))

for i in range(num_of_frames):
    spacetime_F[:, i] = F[:, i]
    spacetime_F_perp[:, i] = np.angle(F_perp)[:, i]
    spacetime_Fz[:, i] = fz[:, i]

# Set up plots
fig, ax = plt.subplots(3, 1, sharex=True)
for axis in ax:
    axis.set_ylabel(r'$x/\xi_s$')
    if axis == ax[0]:
        axis.set_title(r'$|F|$')
    if axis == ax[1]:
        axis.set_title(r'Angle$(F_{perp})$')
    if axis == ax[2]:
        axis.set_title(r'$F_z$')
        axis.set_xlabel(r'$t/\tau$')

# Plot the results
F_plot = ax[0].pcolormesh(time, X, spacetime_F, vmin=0, vmax=1, shading='auto', cmap='jet')
F_perp_plot = ax[1].pcolormesh(time, X, spacetime_F_perp, vmin=-np.pi, vmax=np.pi, shading='auto')
Fz_plot = ax[2].pcolormesh(time, X, spacetime_Fz, vmin=-1, vmax=1, shading='auto')

# Set up cbars
F_cbar = plt.colorbar(F_plot, ax=ax[0])
F_perp_cbar = plt.colorbar(F_perp_plot, ax=ax[1])
F_perp_cbar.set_ticks([-np.pi, 0, np.pi])
F_perp_cbar.set_ticklabels([r'-$\pi$', '0', r'$\pi$'])
Fz_cbar = plt.colorbar(Fz_plot, ax=ax[2])
Fz_cbar.set_ticks([-1, 1])
Fz_cbar.set_ticklabels(['-1', '1'])

plt.show()
