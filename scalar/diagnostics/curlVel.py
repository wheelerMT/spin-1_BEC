import h5py
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def spectral_derivative(array, wvn):
    return ifft2((1j * wvn * (fft2(array))))


# Load the data:
filename = 'scalar_fixed_noRand'
data_path = '../../data/{}.hdf5'.format(filename)

data_file = h5py.File(data_path, 'r')

# Loading grid array data:
x, y = data_file['grid/x'], data_file['grid/y']
X, Y = np.meshgrid(x[:], y[:])
Nx, Ny = x[:].size, y[:].size
dx, dy = x[1] - x[0], y[1] - y[0]

# k-space arrays and meshgrid:
dkx = 2 * np.pi / (Nx * dx)
dky = 2 * np.pi / (Ny * dy)  # K-space spacing
kx = np.fft.fftshift(np.arange(-Nx // 2, Nx // 2) * dkx)
ky = np.fft.fftshift(np.arange(-Ny // 2, Ny // 2) * dky)
Kx, Ky = np.meshgrid(kx, ky)  # K-space meshgrid

psi = data_file['initial_state/psi'][:, :]

# Calculate the mass current:
dpsi_x = spectral_derivative(psi, Kx)
dpsi_y = spectral_derivative(psi, Ky)

nv_x = (np.conj(psi) * dpsi_x - np.conj(dpsi_x) * psi) / (2 * 1j)
nv_y = (np.conj(psi) * dpsi_y - np.conj(dpsi_y) * psi) / (2 * 1j)

dnv_x_y = spectral_derivative(nv_x, Ky)
dnv_y_x = spectral_derivative(nv_y, Kx)

pseudo_vort = -dnv_x_y + dnv_y_x

fig, ax = plt.subplots(1, 2, figsize=(16, 8))
for axis in ax:
    axis.set_aspect('equal')
    axis.set_xlabel(r'$x/\xi$')

ax[0].set_ylabel(r'$y/\xi$')
ax[0].set_title(r'$|\psi|^2$')
ax[1].set_title(r'$\nabla \times n\vec{v}$')

vort_cvals = np.linspace(-500, 500, 100)
dens_plot = ax[0].contourf(X, Y, abs(psi) ** 2, levels=50, cmap='gnuplot')
vort_plot = ax[1].contourf(X, Y, pseudo_vort.real, vort_cvals, cmap='seismic')

dens_cbar = plt.colorbar(dens_plot, ax=ax[0], fraction=0.042)
vort_cbar = plt.colorbar(vort_plot, ax=ax[1], fraction=0.042, ticks=[-500, 0, 500])
vort_cbar.ax.set_yticklabels(['-500', '0', '500'])
# plt.savefig('../../../plots/scalar/psuedoVort_grid_noRand.png')
plt.show()
