import h5py
import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, pi, sin, cos
from matplotlib import cm

"""File for generating spherical harmonics from the wavefunction for a particular dataset."""
# TODO: Fix scaling errors for the spherical harmonics.


# Define harmonic grid:
theta = np.linspace(0, pi, 128)
phi = np.linspace(0, 2 * pi, 128)
Phi, Theta = np.meshgrid(phi, theta)

# Define spherical harmonics:
Y_1p1 = -sqrt(3 / (8 * pi)) * sin(Theta) * np.exp(1j * Phi)
Y_10 = sqrt(3 / (4 * pi)) * cos(Theta)
Y_1m1 = sqrt(3 / (8 * pi)) * sin(Theta) * np.exp(-1j * Phi)

# Load in data:
filename = 'sph_harm_test'  # input('Enter data filename: ')
data_file = h5py.File('../../data/{}.hdf5'.format(filename), 'r')

# Other variables:
x, y = data_file['grid/x'][108:148], data_file['grid/y'][108:148]
dx, dy = x[1] - x[0], y[1] - y[0]
X, Y = np.meshgrid(x, y)

# Wavefunctions:
frame = 1
psi_plus = data_file['wavefunction/psi_plus'][108:148, 108:148, frame]
psi_0 = data_file['wavefunction/psi_0'][108:148, 108:148, frame]
psi_minus = data_file['wavefunction/psi_minus'][108:148, 108:148, frame]

dens = abs(psi_plus) ** 2 + abs(psi_0) ** 2 + abs(psi_minus) ** 2
psi_plus /= np.sqrt(dens)
psi_0 /= np.sqrt(dens)
psi_minus /= np.sqrt(dens)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_axis_off()

scaling = 30

for i in range(0, psi_plus.shape[0], 16):
    for j in range(0, psi_plus.shape[1], 8):
        sph_harm = psi_plus[i, j] * Y_1p1 + psi_0[i, j] * Y_10 + psi_minus[i, j] * Y_1m1
        xx = scaling * abs(sph_harm) ** 2 * sin(Theta) * cos(Phi) + X[i, j]
        yy = scaling * abs(sph_harm) ** 2 * sin(Theta) * sin(Phi) + Y[i, j]
        zz = scaling * abs(sph_harm) ** 2 * cos(Theta)

        ax.plot_surface(xx, yy, zz,  rstride=1, cstride=1, facecolors=cm.binary(np.angle(sph_harm)))

plt.show()
