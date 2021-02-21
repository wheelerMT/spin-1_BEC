import numpy as np
import h5py
import matplotlib.pyplot as plt
import pyfftw
from numpy.fft import fftshift

# --------------------------------------------------------------------------------------------------------------------
# Loading data:
# --------------------------------------------------------------------------------------------------------------------
filename = input('Enter filename of data to open: ')
data_file = h5py.File('../../data/{}.hdf5'.format(filename), 'r')

# Loading grid array data:
x, y = data_file['grid/x'], data_file['grid/y']
X, Y = np.meshgrid(x[:], y[:])
Nx, Ny = x[:].size, y[:].size
dx, dy = x[1] - x[0], y[1] - y[0]
dkx = 2 * np.pi / (Nx * dx)
dky = 2 * np.pi / (Ny * dy)  # K-space spacing
kxx = np.arange(-Nx // 2, Nx // 2) * dkx
kyy = np.arange(-Nx // 2, Nx // 2) * dky
Kx, Ky = np.meshgrid(kxx, kyy)
# Kx, Ky = np.fft.fftshift(Kx), np.fft.fftshift(Ky)
K = np.sqrt(fftshift(Kx ** 2) + fftshift(Ky ** 2))
wvn = K[:Nx//2, 0]

psi = data_file['wavefunction/psi']

num_of_frames = psi.shape[-1]

# Initialising FFTs
wfn_data = pyfftw.empty_aligned((Nx, Ny), dtype='complex64')
fft2 = pyfftw.builders.fft2(wfn_data, threads=8)
ifft2 = pyfftw.builders.ifft2(wfn_data, threads=8)

# --------------------------------------------------------------------------------------------------------------------
# Spectra:
# --------------------------------------------------------------------------------------------------------------------
# Occupation number:
occupation = np.empty((Nx, Ny, num_of_frames), dtype='float32')
for i in range(num_of_frames):
    occupation[:, :, i] = fftshift((fft2(psi[:, :, i])) * (np.conjugate(fft2(psi[:, :, i])))).real / (Nx * Ny)

print('{:.1e}'.format(dkx * dky * np.sum(occupation[:, :, 0] / (dkx * dky))))

# ---------------------------------------------------------------------------------------------------------------------
# Calculating energy spectrum
# ---------------------------------------------------------------------------------------------------------------------
print('Calculating energy spectrum...')
box_radius = int(np.ceil(np.sqrt(Nx ** 2 + Ny ** 2) / 2) + 1)

centerx = Nx // 2
centery = Ny // 2

eps = 1e-50  # Voids log(0)

# Defining zero arrays for spectra:
nk = np.zeros((box_radius, num_of_frames))
nc = np.zeros((box_radius, num_of_frames))  # Counts the number of times we sum over a given shell

# Summing over spherical shells:
for index in range(1):
    for kx in range(Nx):
        for ky in range(Ny):
            k = int(np.round(np.sqrt((kx - centerx) ** 2 + (ky - centery) ** 2)))   # Radius of shell
            nc[k, index] += 1

            nk[k, index] += occupation[kx, ky, index] # / np.sqrt(Kx[kx, ky] ** 2 + Ky[kx, ky] ** 2)  # n(k)

    print('On index {}.'.format(index))

for i in range(1):
    nk[:, i] /= (nc[:, i] * dkx)

print('{:.1e}'.format(np.sum(2 * np.pi * nk[1:Nx // 2, 0] * wvn[1:])))

fig, ax = plt.subplots(1, )
ax.set_xlabel(r'$k a_s$')
ax.set_ylabel(r'$n(k)$')
plt.loglog(wvn, nk[:Nx // 2, 0], 'kD', markersize=2, label=r'$\tau=3.0\times10^3$')

plt.legend()
# plt.savefig('../../../plots/scalar/Gasenzer_occ_num.png')
plt.show()
