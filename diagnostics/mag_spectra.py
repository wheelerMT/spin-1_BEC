import h5py
import numpy as np
import numexpr as ne
from numpy import conj
from numpy.fft import fftshift, fft2
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


filename = input('Enter filename to load data: ')
data_file = h5py.File('../data/{}.hdf5'.format(filename), 'r')

# Grid data:
x, y = np.array(data_file['grid/x']), np.array(data_file['grid/y'])
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
total_frames = data_file['num_of_frames'][...]
saved_times = data_file['saved_times']

n_occ_plus = np.empty((Nx, Ny, num_of_frames), dtype='float32')
n_occ_0 = np.empty((Nx, Ny, num_of_frames), dtype='float32')
n_occ_minus = np.empty((Nx, Ny, num_of_frames), dtype='float32')
for i in range(num_of_frames):
    n_occ_plus[:, :, i] = fftshift(fft2(psi_plus[:, :, i]) * conj(fft2(psi_plus[:, :, i]))).real / (Nx * Ny)
    n_occ_0[:, :, i] = fftshift(fft2(psi_0[:, :, i]) * conj(fft2(psi_0[:, :, i]))).real / (Nx * Ny)
    n_occ_minus[:, :, i] = fftshift(fft2(psi_minus[:, :, i]) * conj(fft2(psi_minus[:, :, i]))).real / (Nx * Ny)

# ---------------------------------------------------------------------------------------------------------------------
# Calculating energy spectrum
# ---------------------------------------------------------------------------------------------------------------------
box_radius = int(np.ceil(np.sqrt(Nx ** 2 + Ny ** 2) / 2) + 1)

centerx = Nx // 2
centery = Ny // 2

eps = 1e-50  # Voids log(0)

# Defining zero arrays for spectra:
e_occ_plus = np.zeros((box_radius, num_of_frames))
e_occ_0 = np.zeros((box_radius, num_of_frames))
e_occ_minus = np.zeros((box_radius, num_of_frames))

nc = np.zeros((box_radius, num_of_frames))  # Counts the number of times we sum over a given shell

list_of_times = []
for i in range(psi_plus.shape[-1]):
    list_of_times.append([i, saved_times[i]])
print(tabulate(list_of_times, headers=["Frame #", "Time"], tablefmt="orgtbl"))

frame = int(input('Enter the index of the frame you wish to use: '))  # Array index

# Summing over spherical shells:
for kx in range(Nx):
    for ky in range(Ny):
        k = int(np.ceil(np.sqrt((kx - centerx) ** 2 + (ky - centery) ** 2)))
        nc[k, frame] += 1

        e_occ_plus[k, frame] += n_occ_plus[kx, ky, frame]
        e_occ_0[k, frame] += n_occ_0[kx, ky, frame]
        e_occ_minus[k, frame] += n_occ_minus[kx, ky, frame]


e_occ_plus[:, frame] /= (nc[:, frame] * dkx)
e_occ_0[:, frame] /= (nc[:, frame] * dkx)
e_occ_minus[:, frame] /= (nc[:, frame] * dkx)

# Plots:
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# Occupation number:
axes[0].loglog(wvn, e_occ_plus[:Nx // 2, frame], color='b', marker='D', markersize=3, linestyle='None', label='$n_+(k)$')
axes[0].loglog(wvn, e_occ_0[:Nx // 2, frame], color='r', marker='D', markersize=3, linestyle='None', label='$n_0(k)$')
axes[0].loglog(wvn, e_occ_minus[:Nx // 2, frame], color='g', marker='D', markersize=3, linestyle='None', label='$n_-(k)$')
axes[0].loglog(wvn, e_occ_minus[:Nx // 2, frame] + e_occ_0[:Nx // 2, frame] + e_occ_plus[:Nx // 2, frame],
               color='k', marker='D', markersize=3, linestyle='None', label=r'$\Sigma n_\delta(k)$')

# Plot k-lines:
# axes[0].loglog(wvn[40:], 2.5e4 * wvn[40:] ** (-2), 'k', label=r'$k^{-2}$')
# axes[0].loglog(wvn[3:30], 7e2 * wvn[3:30] ** (-3), 'k--', label=r'$k^{-3}$')
# axes[0].loglog(wvn[1:30], 6e2 * wvn[1:30] ** (-4), 'k:', label=r'$k^{-4}$')

# Plot n_+ - n_-:
axes[1].semilogx(wvn, e_occ_plus[:Nx // 2, frame] - e_occ_minus[:Nx // 2, frame], 'k', marker='D', markersize=3,
                 linestyle='None', label=r'$\tau={}$'.format(saved_times[frame]))

for ax in axes:
    ax.set_xlabel(r'$k\xi$')
    ax.set_aspect('auto')

    # Titles:
    if ax == axes[0]:
        ax.set_ylabel(r'$n(k)$')
        ax.set_title(r'$\tau={}$'.format(saved_times[frame]))
        ax.set_ylim(top=4e8, bottom=1e0)
    if ax == axes[1]:
        ax.set_ylabel(r'$n_+(k) - n_-(k)$')
        ax.set_ylim(top=5e8, bottom=-3e8)

axes[0].legend(loc=3)
axes[1].legend(loc=1)
save_image = input('Do you want to save the image? (y/n): ')
if save_image == 'y':
    image_name = input('Enter name of image to be saved: ')
    plt.savefig('../images/unsorted/{}.png'.format(image_name), dpi=200)

plt.show()
