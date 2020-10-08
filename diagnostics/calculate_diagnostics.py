import pyfftw
import progressbar
import h5py
import numpy as np
import include.diag as diag

# ---------------------------------------------------------------------------------------------------------------------
# Loading data
# ---------------------------------------------------------------------------------------------------------------------
filename = input('Enter name of data file: ')
data_file = h5py.File('../data/{}.hdf5'.format(filename), 'r+')
diag_file = h5py.File('../data/diagnostics/{}_diag.hdf5'.format(filename), 'w')  # Diagnostics file to save to

# Loading grid array data:
x, y = data_file['grid/x'], data_file['grid/y']
X, Y = np.meshgrid(x, y)
Nx, Ny = x.size, y.size
dx, dy = x[1] - x[0], y[1] - y[0]
dkx, dky = 2 * np.pi / (Nx * dx), 2 * np.pi / (Ny * dy)
kxx = np.arange(-Nx // 2, Nx // 2) * dkx
kyy = np.arange(-Nx // 2, Nx // 2) * dky
Kx, Ky = np.meshgrid(kxx, kyy)

# Wavefunction data
psi_plus = data_file['wavefunction/psi_plus']
psi_0 = data_file['wavefunction/psi_0']
psi_minus = data_file['wavefunction/psi_minus']

# Number of frames:
num_of_frames = psi_plus.shape[-1]
print('We are working with %i frames of data.' % num_of_frames)

# Progress bar widgets:
widgets = [' [', progressbar.Timer(), '] ', progressbar.Bar(), ' (', progressbar.ETA(), ') ']

# FFT setup:
fft_data = pyfftw.empty_aligned((Nx, Ny), dtype='complex64')
fft2 = pyfftw.builders.fft2(fft_data, threads=8)
ifft2 = pyfftw.builders.ifft2(fft_data, threads=8)

# ---------------------------------------------------------------------------------------------------------------------
# Calculating quantities
# ---------------------------------------------------------------------------------------------------------------------
# Total density
print('Calculating density...')
n = np.empty((Nx, Ny, num_of_frames), dtype='float32')

with progressbar.ProgressBar(max_value=num_of_frames, widgets=widgets) as bar:
    for i in bar(range(num_of_frames)):
        n[:, :, i] = diag.calculate_density(psi_plus[:, :, i], psi_0[:, :, i], psi_minus[:, :, i])

        bar.update(bar.value)

# Spin vector & expectation
print('Calculating spin vectors...')
Fx = diag_file.create_dataset('spin/Fx', (Nx, Ny, num_of_frames), dtype='float32')
Fy = diag_file.create_dataset('spin/Fy', (Nx, Ny, num_of_frames), dtype='float32')
Fz = diag_file.create_dataset('spin/Fz', (Nx, Ny, num_of_frames), dtype='float32')
spin_expec_mag = diag_file.create_dataset('spin/spin_expectation', (Nx, Ny, num_of_frames), dtype='float32')

with progressbar.ProgressBar(max_value=num_of_frames, widgets=widgets) as bar:
    for i in bar(range(num_of_frames)):
        Fx[:, :, i], Fy[:, :, i], Fz[:, :, i], spin_expec_mag[:, :, i] = \
            diag.calculate_spin(psi_plus[:, :, i], psi_0[:, :, i], psi_minus[:, :, i], n[:, :, i])

        bar.update(bar.value)

# Gradients:
print('Calculating spectral gradients...')
grad_p_x, grad_p_y = np.empty((Nx, Ny, num_of_frames), dtype='complex64'), np.empty((Nx, Ny, num_of_frames),
                                                                                    dtype='complex64')
grad_0_x, grad_0_y = np.empty((Nx, Ny, num_of_frames), dtype='complex64'), np.empty((Nx, Ny, num_of_frames),
                                                                                    dtype='complex64')
grad_m_x, grad_m_y = np.empty((Nx, Ny, num_of_frames), dtype='complex64'), np.empty((Nx, Ny, num_of_frames),
                                                                                    dtype='complex64')

with progressbar.ProgressBar(max_value=num_of_frames, widgets=widgets) as bar:
    for i in range(num_of_frames):
        grad_p_x[:, :, i], grad_p_y[:, :, i] = diag.spectral_derivative(psi_plus[:, :, i], Kx, Ky, fft2, ifft2)
        grad_0_x[:, :, i], grad_0_y[:, :, i] = diag.spectral_derivative(psi_0[:, :, i], Kx, Ky, fft2, ifft2)
        grad_m_x[:, :, i], grad_m_y[:, :, i] = diag.spectral_derivative(psi_minus[:, :, i], Kx, Ky, fft2, ifft2)
        bar.update(bar.value)

# Mass current:
print('Calculating mass current...')
nv_mass_x = diag_file.create_dataset('velocity/nv_mass_x', (Nx, Ny, num_of_frames), dtype='float32')
nv_mass_y = diag_file.create_dataset('velocity/nv_mass_y', (Nx, Ny, num_of_frames), dtype='float32')
pseudo_vorticity = diag_file.create_dataset('velocity/pseudo_vorticity', (Nx, Ny, num_of_frames), dtype='float32')

with progressbar.ProgressBar(max_value=num_of_frames, widgets=widgets) as bar:
    for i in bar(range(num_of_frames)):
        nv_mass_x[:, :, i], nv_mass_y[:, :, i] = \
            diag.calculate_mass_current(psi_plus[:, :, i], psi_0[:, :, i], psi_minus[:, :, i],
                                        grad_p_x[:, :, i], grad_p_y[:, :, i], grad_0_x[:, :, i], grad_0_y[:, :, i],
                                        grad_m_x[:, :, i], grad_m_y[:, :, i])
        pseudo_vorticity[:, :, i] = diag.calculate_pseudo_vorticity(nv_mass_x[:, :, i], nv_mass_y[:, :, i], dx, dy)

        bar.update(bar.value)

data_file.close()
diag_file.close()
