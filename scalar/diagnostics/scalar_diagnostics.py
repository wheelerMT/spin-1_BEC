import numpy as np
import h5py
import numexpr as ne
import pyfftw
from numpy.fft import fftshift, ifftshift

# Load in data file:
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

# Load in wavefunction:
psi = data_file['wavefunction/psi']
num_of_frames = psi.shape[-1]

# Initialising FFTs
wfn_data = pyfftw.empty_aligned((Nx, Ny), dtype='complex64')
fft2 = pyfftw.builders.fft2(wfn_data, threads=8)
ifft2 = pyfftw.builders.ifft2(wfn_data, threads=8)

# Save desired quantities to file:
saved_data = h5py.File('../../data/diagnostics/{}_diag.hdf5'.format(filename), 'w')

# ----------------------------------------------------------------------------------------------------------------------
# Calculate quantities
# ----------------------------------------------------------------------------------------------------------------------
# Density:
print('Calculating density...')
n = np.empty((Nx, Ny, num_of_frames))
for i in range(num_of_frames):
    n[:, :, i] = abs(psi[:, :, i]) ** 2
    if np.mod(i, num_of_frames // 4) == 0:
        print('Calculating density: %i%% complete. ' % (i / num_of_frames * 100))

# Spectral derivatives:
print('Calculating spectral derivatives...')
grad_x = grad_y = np.empty((Nx, Ny, num_of_frames), dtype='complex64')
for i in range(num_of_frames):
    grad_x[:, :, i] = ifft2(ifftshift(1j * Kx * fftshift(fft2(psi[:, :, i]))))  # d / dx
    grad_y[:, :, i] = ifft2(ifftshift(1j * Ky * fftshift(fft2(psi[:, :, i]))))  # d / dy
    if np.mod(i, num_of_frames // 4) == 0:
        print('Calculating spectral derivatives: %i%% complete. ' % (i / num_of_frames * 100))

grad_x_c, grad_y_c = np.conjugate(grad_x), np.conjugate(grad_y)

# Mass current:
print('Calculating mass current...')
nv_mass_x = nv_mass_y = np.empty((Nx, Ny, num_of_frames), dtype='float32')
v_mass_x = saved_data.create_dataset('velocity/v_mass_x', (Nx, Ny, num_of_frames), dtype='float32')
v_mass_y = saved_data.create_dataset('velocity/v_mass_y', (Nx, Ny, num_of_frames), dtype='float32')

for i in range(num_of_frames):
    psi2d = psi[:, :, i]
    grad_x2d = grad_x[:, :, i]
    grad_y2d = grad_y[:, :, i]

    nv_mass_x[:, :, i] = ne.evaluate("(conj(psi2d) * grad_x2d - conj(grad_x2d) * psi2d) / 2j").real
    nv_mass_y[:, :, i] = ne.evaluate("(conj(psi2d) * grad_y2d - conj(grad_y2d) * psi2d) / 2j").real
    v_mass_x[:, :, i] = nv_mass_x[:, :, i] / n[:, :, i]
    v_mass_y[:, :, i] = nv_mass_y[:, :, i] / n[:, :, i]
    if np.mod(i, num_of_frames // 4) == 0:
        print('Calculating mass current: %i%% complete. ' % (i / num_of_frames * 100))

# Pseudo-vorticity:
print('Calculating pseudo-vorticity...')
curl_nv_mass = saved_data.create_dataset('velocity/curl_nv_mass', (Nx, Ny, num_of_frames), dtype='float32')
for i in range(num_of_frames):
    curl_nv_mass[:, :, i] = (ifft2(ifftshift(1j * Kx * fftshift(fft2(nv_mass_y[:, :, i]))))
                             - ifft2(ifftshift(1j * Ky * fftshift(fft2(nv_mass_x[:, :, i]))))).real
    if np.mod(i, num_of_frames // 4) == 0:
        print('Calculating pseudo-vorticity: %i%% complete. ' % (i / num_of_frames * 100))

saved_data.close()
