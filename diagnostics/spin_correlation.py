import h5py
import numpy as np
import pyfftw
import numexpr as ne
import include.diag as diag
from numpy.fft import fftshift
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# ------------------------------------------------------------------------------------------------------------------
# Loading required data
# ------------------------------------------------------------------------------------------------------------------
filename = input('Enter name of data file: ')
data_file = h5py.File('../data/{}.hdf5'.format(filename), 'r')
diag_file = h5py.File('../data/diagnostics/{}_diag.hdf5'.format(filename), 'r')

# Grid data:
x, y = np.array(data_file['grid/x']), np.array(data_file['grid/y'])
X, Y = np.meshgrid(x, y)
Nx, Ny = x.size, y.size
dx, dy = x[1] - x[0], y[1] - y[0]
dkx = 2 * np.pi / (Nx * dx)
dky = 2 * np.pi / (Ny * dy)  # K-space spacing
kxx = np.arange(-Nx // 2, Nx // 2) * dkx
kyy = np.arange(-Nx // 2, Nx // 2) * dky
Kx, Ky = np.meshgrid(kxx, kyy)
K = ne.evaluate("sqrt(Kx ** 2 + Ky ** 2)")
wvn = kxx[Nx // 2:]

# Three component wavefunction
psi_plus = data_file['wavefunction/psi_plus']
psi_0 = data_file['wavefunction/psi_0']
psi_minus = data_file['wavefunction/psi_minus']

num_of_frames = psi_plus.shape[-1]

wfn_data = pyfftw.empty_aligned((Nx, Ny), dtype='complex64')
fft2 = pyfftw.builders.fft2(wfn_data)
ifft2 = pyfftw.builders.ifft2(wfn_data)

# Get info about the saved times:
saved_times = data_file['saved_times']

# ------------------------------------------------------------------------------------------------------------------
# Calculating density and spin vectors:
# ------------------------------------------------------------------------------------------------------------------
# Density:
print('Calculating density...')
n = np.empty((Nx, Ny, num_of_frames), dtype='float32')
for i in range(num_of_frames):
    n[:, :, i] = diag.calculate_density(psi_plus[:, :, i], psi_0[:, :, i], psi_minus[:, :, i])

Fx, Fy, Fz = diag_file['spin/Fx'], diag_file['spin/Fy'], diag_file['spin/Fz']
F_perp = Fx[...] + 1j * Fy[...]

ws_x = np.empty((Nx, Ny, num_of_frames), dtype='complex64')
ws_y = np.empty((Nx, Ny, num_of_frames), dtype='complex64')
ws_z = np.empty((Nx, Ny, num_of_frames), dtype='complex64')
ws_perp = np.empty((Nx, Ny, num_of_frames), dtype='complex64')

# Generalised spin velocity:
for i in range(num_of_frames):
    ws_x[:, :, i] = fftshift(fft2(Fx[:, :, i] / n[:, :, i])) / np.sqrt(Nx * Ny)

    ws_y[:, :, i] = fftshift(fft2(Fy[:, :, i] / n[:, :, i])) / np.sqrt(Nx * Ny)

    ws_z[:, :, i] = fftshift(fft2(Fz[:, :, i] / n[:, :, i])) / np.sqrt(Nx * Ny)

    ws_perp[:, :, i] = fftshift(fft2(F_perp[:, :, i] / n[:, :, i])) / np.sqrt(Nx * Ny)

# Calculating respective energies:
# Spin:
E_s_x = ne.evaluate("abs(ws_x).real ** 2")
E_s_y = ne.evaluate("abs(ws_y).real ** 2")
E_s_z = ne.evaluate("abs(ws_z).real ** 2")
E_s_perp = ne.evaluate("abs(ws_perp).real ** 2")

# ---------------------------------------------------------------------------------------------------------------------
# Calculating energy spectrum
# ---------------------------------------------------------------------------------------------------------------------
box_radius = int(np.ceil(np.sqrt(Nx ** 2 + Ny ** 2) / 2) + 1)

centerx = Nx // 2
centery = Ny // 2

eps = 1e-50  # Voids log(0)

# Defining zero arrays for spectra:
e_s_x = np.zeros((box_radius, num_of_frames)) + eps
e_s_y = np.zeros((box_radius, num_of_frames)) + eps
e_s_z = np.zeros((box_radius, num_of_frames)) + eps
e_s_perp = np.zeros((box_radius, num_of_frames)) + eps
nc = np.zeros((box_radius, num_of_frames))  # Counts the number of times we sum over a given shell

# Summing over spherical shells:
print('Summing over spherical shells...')
for index in range(num_of_frames):
    for kx in range(Nx):
        for ky in range(Ny):
            k = int(np.ceil(np.sqrt((kx - centerx) ** 2 + (ky - centery) ** 2)))
            nc[k, index] += 1

            e_s_x[k, index] += E_s_x[kx, ky, index]
            e_s_y[k, index] += E_s_y[kx, ky, index]
            e_s_z[k, index] += E_s_z[kx, ky, index]
            e_s_perp[k, index] += E_s_perp[kx, ky, index]

    print('On frame1: %i' % (index + 1))

for i in range(num_of_frames):
    e_s_x[:, i] /= (nc[:, i] * dkx)
    e_s_y[:, i] /= (nc[:, i] * dkx)
    e_s_z[:, i] /= (nc[:, i] * dkx)
    e_s_perp[:, i] /= (nc[:, i] * dkx)

fig, axes = plt.subplots(1, figsize=(8, 8))
axes.set_xlabel(r'$k\xi$')
axes.set_ylim(top=4e4, bottom=1e-2)
axes.set_ylabel(r'$S_\perp(k)$')
colors = ['r', 'g', 'b', 'y', 'c', 'k']

for i in range(1, num_of_frames):
    # Occupation number:
    axes.loglog(wvn, e_s_perp[:Nx // 2, i], color=colors[i], marker='D', markersize=3,
                linestyle='None', label=r'$\tau = {}$'.format(saved_times[i]))

axes.loglog(wvn[5:120], 1.5e1 * wvn[5:120] ** (-7 / 3), 'k-', label=r'$k^{-7/3}$')
axes.loglog(wvn[5:120], 0.5e1 * wvn[5:120] ** (-2), 'k--', label=r'$k^{-2}$')
axes.legend(loc=3)

plt.show()
