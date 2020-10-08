import matplotlib.pyplot as plt
import h5py
import numpy as np


Mx = My = 64
Nx = Ny = 128  # Number of grid pts
dx = dy = 1 / 2  # Grid spacing
dkx = np.pi / (Mx * dx)
dky = np.pi / (My * dy)  # K-space spacing
len_x = Nx * dx     # Box length
len_y = Ny * dy
x = np.arange(-Mx, Mx) * dx
y = np.arange(-My, My) * dy
X, Y = np.meshgrid(x, y)  # Spatial meshgrid

data_plots = h5py.File('../data/scalar_data.hdf5', 'a')
psi_plots = np.array(data_plots['wavefunction/psi'])

fig, ax = plt.subplots(1, figsize=(8, 8))
ax.set_xlabel(r'$x / \xi$')
ax.set_ylabel(r'$y / \xi$')
ax.set_aspect('equal')
initial_cont = ax.contourf(X, Y, abs(psi_plots[:, :, 0])**2, cmap='gnuplot', levels=50)
cbar = fig.colorbar(initial_cont, shrink=0.85)
for i in range(np.ma.size(psi_plots, -1)):
    ax.contourf(X, Y, abs(psi_plots[:, :, i]) ** 2, levels=50, cmap='gnuplot')
    plt.title(r'$\tau = %03f$' % (i * 5e-3 * 200))
    plt.savefig('images_scalar/dens%03d.png' % i)
    print('Figure {} saved.'.format(i))
