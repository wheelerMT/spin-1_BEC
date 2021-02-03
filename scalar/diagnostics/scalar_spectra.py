import numpy as np
import h5py
import matplotlib.pyplot as plt
import numexpr as ne
import pyfftw
from numpy.fft import fftshift, ifftshift


# --------------------------------------------------------------------------------------------------------------------
# Loading data:
# --------------------------------------------------------------------------------------------------------------------
filename = input('Enter filename of data to open: ')
data_file = h5py.File('../../data/{}.hdf5'.format(filename), 'r')
diagnostics_file = h5py.File('../../data/diagnostics/{}_diag.hdf5'.format(filename), 'r')

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
K = ne.evaluate("sqrt(Kx ** 2 + Ky ** 2)")
wvn = kxx[Nx//2:]

psi = data_file['wavefunction/psi']

num_of_frames = psi.shape[-1]

# Initialising FFTs
wfn_data = pyfftw.empty_aligned((Nx, Ny), dtype='complex64')
fft2 = pyfftw.builders.fft2(wfn_data, threads=8)
ifft2 = pyfftw.builders.ifft2(wfn_data, threads=8)

# --------------------------------------------------------------------------------------------------------------------
# Calculating density and velocity:
# --------------------------------------------------------------------------------------------------------------------
# Density
print('Calculating density...')
n = np.empty((Nx, Ny, num_of_frames))
for i in range(num_of_frames):
    n[:, :, i] = abs(psi[:, :, i]) ** 2
    if np.mod(i, num_of_frames // 4) == 0:
        print('Calculating density: %i%% complete. ' % (i / num_of_frames * 100))

# Mass velocity:
v_mass_x = diagnostics_file['velocity/v_mass_x']
v_mass_y = diagnostics_file['velocity/v_mass_y']

# --------------------------------------------------------------------------------------------------------------------
# Spectra:
# --------------------------------------------------------------------------------------------------------------------
# Occupation number:
occupation = np.empty((Nx, Ny, num_of_frames), dtype='float32')
for i in range(num_of_frames):
    occupation[:, :, i] = 0.5 * fftshift((fft2(psi[:, :, i])) * (np.conjugate(fft2(psi[:, :, i])))).real / (Nx * Ny)

# Coefficients of incompressible and compressible velocities:
A_1 = ne.evaluate("1 - Kx ** 2 / K ** 2")
A_2 = ne.evaluate("1 - Ky ** 2 / K ** 2")
B = ne.evaluate("Kx * Ky / K ** 2")
C_1 = ne.evaluate("Kx ** 2 / K ** 2")
C_2 = ne.evaluate("Ky ** 2 / K ** 2")

# Quantum pressure:
grad_sqrtn_x = np.empty((Nx, Ny, num_of_frames), dtype='complex64')
grad_sqrtn_y = np.empty((Nx, Ny, num_of_frames), dtype='complex64')
wq_x = np.empty((Nx, Ny, num_of_frames), dtype='complex64')
wq_y = np.empty((Nx, Ny, num_of_frames), dtype='complex64')
print('Calculating quantum pressure term...')
for i in range(num_of_frames):
    grad_sqrtn_x[:, :, i] = ifft2(ifftshift(1j * Kx * fftshift(fft2(np.sqrt(n[:, :, i])))))  # d / dx
    grad_sqrtn_y[:, :, i] = ifft2(ifftshift(1j * Ky * fftshift(fft2(np.sqrt(n[:, :, i])))))  # d / dy

    wq_x[:, :, i] = fftshift(fft2(grad_sqrtn_x[:, :, i])) / np.sqrt(Nx * Ny)
    wq_y[:, :, i] = fftshift(fft2(grad_sqrtn_y[:, :, i])) / np.sqrt(Nx * Ny)
    if np.mod(i, num_of_frames // 4) == 0:
        print('Calculating quantum pressure: %i%% complete. ' % (i / num_of_frames * 100))

# Calculating Fourier transform of mass velocity:
u_x = u_y = np.empty((Nx, Ny, num_of_frames), dtype='complex64')
print('Calculating Fourier transform of mass velocity...')
for i in range(num_of_frames):
    u_x[:, :, i] = fftshift(fft2(np.sqrt(n[:, :, i]) * v_mass_x[:, :, i])) / np.sqrt(Nx * Ny)
    u_y[:, :, i] = fftshift(fft2(np.sqrt(n[:, :, i]) * v_mass_y[:, :, i])) / np.sqrt(Nx * Ny)
    if np.mod(i, num_of_frames // 4) == 0:
        print('Calculating weighted velocity: %i%% complete. ' % (i / num_of_frames * 100))

# Incompressible:
ui_x = ui_y = np.empty((Nx, Ny, num_of_frames), dtype='complex64')
print('Calculating incompressible term...')
for i in range(num_of_frames):
    u_xx = u_x[:, :, i]
    u_yy = u_y[:, :, i]
    ui_x[:, :, i] = ne.evaluate("A_1 * u_xx - B * u_yy")
    ui_y[:, :, i] = ne.evaluate("-B * u_xx + A_2 * u_yy")
    if np.mod(i, num_of_frames // 4) == 0:
        print('Calculating incompressible velocity: %i%% complete. ' % (i / num_of_frames * 100))

# Compressible:
uc_x = uc_y = np.empty((Nx, Ny, num_of_frames), dtype='complex64')
print('Calculating compressible term...')
for i in range(num_of_frames):
    u_xx = u_x[:, :, i]
    u_yy = u_y[:, :, i]
    uc_x[:, :, i] = ne.evaluate("C_1 * u_xx + B * u_yy")
    uc_y[:, :, i] = ne.evaluate("B * u_xx + C_2 * u_yy")
    if np.mod(i, num_of_frames // 4) == 0:
        print('Calculating compressible velocity: %i%% complete. ' % (i / num_of_frames * 100))

data_file.close()
diagnostics_file.close()

# ---------------------------------------------------------------------------------------------------------------------
# Calculating energies
# ---------------------------------------------------------------------------------------------------------------------
print('Calculating energies...')
# Quantum pressure:
E_q = ne.evaluate("0.5 * (abs(wq_x).real ** 2 + abs(wq_y).real ** 2)")
print('Quantum pressure energy calculated.')

# Total velocity
E_v_tot = ne.evaluate("0.5 * (abs(u_x).real ** 2 + abs(u_y).real ** 2)")
print('Total velocity energy calculated.')

# Incompressible:
E_vi = ne.evaluate("0.5 * (abs(ui_x).real ** 2 + abs(ui_y).real ** 2)")
print('Incompressible energy calculated.')

# Compressible:
E_vc = ne.evaluate("0.5 * (abs(uc_x).real ** 2 + abs(uc_y).real ** 2)")
print('Compressible energy calculated.')

# ---------------------------------------------------------------------------------------------------------------------
# Calculating energy spectrum
# ---------------------------------------------------------------------------------------------------------------------
print('Calculating energy spectrum...')
box_radius = int(np.ceil(np.sqrt(Nx ** 2 + Ny ** 2) / 2) + 1)

centerx = Nx // 2
centery = Ny // 2

# Open spectra file to save spectra data:
spectra_file = h5py.File('../../data/spectra/{}_spectra.hdf5'.format(filename), 'w')
spectra_file.create_dataset('wvn', data=wvn)    # Saves wavenumbers

# Defining datasets for spectra:
e_occ = spectra_file.create_dataset('e_occ', (box_radius, num_of_frames), dtype='float32')
e_q = spectra_file.create_dataset('e_q', (box_radius, num_of_frames), dtype='float32')
e_vi = spectra_file.create_dataset('e_vi', (box_radius, num_of_frames), dtype='float32')
e_vc = spectra_file.create_dataset('e_vc', (box_radius, num_of_frames), dtype='float32')
nc = np.zeros((box_radius, num_of_frames))

# Defining zero arrays for calculating spectra:
e_occ_calc = np.zeros((box_radius, num_of_frames), dtype='float32')
e_q_calc = np.zeros((box_radius, num_of_frames), dtype='float32')
e_vi_calc = np.zeros((box_radius, num_of_frames), dtype='float32')
e_vc_calc = np.zeros((box_radius, num_of_frames), dtype='float32')

# Summing over spherical shells:
for index in range(num_of_frames):
    for kx in range(Nx):
        for ky in range(Ny):
            k = int(np.round(np.sqrt((kx - centerx) ** 2 + (ky - centery) ** 2)))
            nc[k, index] += 1
            e_occ_calc[k, index] = e_occ_calc[k, index] + occupation[kx, ky, index]
            e_q_calc[k, index] = e_q_calc[k, index] + K[kx, ky] ** (-2) * E_q[kx, ky, index]
            e_vi_calc[k, index] = e_vi_calc[k, index] + K[kx, ky] ** (-2) * E_vi[kx, ky, index]
            e_vc_calc[k, index] = e_vc_calc[k, index] + K[kx, ky] ** (-2) * E_vc[kx, ky, index]

    e_occ[:, index] = e_occ_calc[:, index] / (nc[:, index] * dkx)
    e_q[:, index] = e_q_calc[:, index] / (nc[:, index] * dkx)
    e_vi[:, index] = e_vi_calc[:, index] / (nc[:, index] * dkx)
    e_vc[:, index] = e_vc_calc[:, index] / (nc[:, index] * dkx)
    print('On spectra frame: %i' % (index + 1))
