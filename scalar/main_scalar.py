import numpy as np
from numpy import pi, exp
import h5py
import pyfftw
import numexpr as ne

# --------------------------------------------------------------------------------------------------------------------
# Controlled variables:
# --------------------------------------------------------------------------------------------------------------------
Mx = My = 256
Nx = Ny = 512  # Number of grid pts
dx = dy = 1  # Grid spacing
dkx = pi / (Mx * dx)
dky = pi / (My * dy)  # K-space spacing
len_x = Nx * dx  # Box length
len_y = Ny * dy
x = np.arange(-Mx, Mx) * dx
y = np.arange(-My, My) * dy
X, Y = np.meshgrid(x, y)  # Spatial meshgrid

# k-space arrays and meshgrid:
kx = np.fft.fftshift(np.arange(-Mx, Mx) * dkx)
ky = np.fft.fftshift(np.arange(-My, My) * dky)
Kx, Ky = np.meshgrid(kx, ky)  # K-space meshgrid

# Initialising FFTs
wfn_data = pyfftw.empty_aligned((Nx, Ny), dtype='complex128')
fft_forward = pyfftw.FFTW(wfn_data, wfn_data, axes=(0, 1), threads=8)
fft_backward = pyfftw.FFTW(wfn_data, wfn_data, direction='FFTW_BACKWARD', axes=(0, 1), threads=8)

# Aligning empty array to speed up FFT:
psi = pyfftw.empty_aligned((Nx, Ny), dtype='complex128')

# Controlled variables
V = 0.  # Doubly periodic box
c0 = 1e-3

# Time steps, number and wavefunction save variables
Nt = 4000000
Nframe = 2500   # Save data every Nframe number of timesteps
dt = 1e-2  # Imaginary time timestep
t = 0.
k = 0   # Array index

filename = 'scalar'    # Name of file to save data to
data_path = 'data/{}.hdf5'.format(filename)
backup_data_path = 'data/{}_backup.hdf5'.format(filename)

fresh_simulation = True  # Boolean that corresponds to a fresh simulation if True or a continued simulation if False

# --------------------------------------------------------------------------------------------------------------------
# Generating initial state:
# --------------------------------------------------------------------------------------------------------------------
# If it is a continued simulation, load the previous data and continue to evolution:
if not fresh_simulation:
    previous_data = h5py.File(backup_data_path, 'r')
    psi_k = np.array(previous_data['wavefunction/psi_k'])
    t = np.round(previous_data['time'][...])
    k = previous_data['array_index'][...]
    previous_data.close()

# If it is a fresh simulation, generate the initial state:
else:
    atom_num = 1e8 / (Nx * Ny)

    # Draw values of theta from Gaussian distribution:
    np.random.seed(9999)
    theta_k = np.random.uniform(low=0, high=1, size=(Nx, Ny)) * 2 * np.pi

    # Generate density array so that only certain modes are occupied:
    n_k = np.zeros((Nx, Ny))
    n_k[Nx // 2, Ny // 2] = atom_num * 0.5 / (dx * dy)    # k = (0, 0)
    n_k[Nx // 2 + 1, Ny // 2] = atom_num * 0.125 / (dx * dy)  # k = (1, 0)
    n_k[Nx // 2 + 1, Ny // 2 + 1] = atom_num * 0.125 / (dx * dy)  # k = (1, 1)
    n_k[Nx // 2 - 1, Ny // 2] = atom_num * 0.125 / (dx * dy)  # k = (-1, 0)
    n_k[Nx // 2 - 1, Ny // 2 - 1] = atom_num * 0.125 / (dx * dy)  # k = (-1, -1)

    # Construct wavefunction in Fourier space:
    psi_k = np.fft.fftshift(Nx * Ny * np.sqrt(n_k) * exp(1j * theta_k))
    pyfftw.byte_align(psi_k)    # Aligns array to speed up FFT

    # Creating file to save to:
    with h5py.File(data_path, 'w') as data:
        # Saving spatial data:
        data.create_dataset('grid/x', x.shape, data=x)
        data.create_dataset('grid/y', y.shape, data=y)

        # Saving time variables:
        data.create_dataset('time/Nt', data=Nt)
        data.create_dataset('time/dt', data=dt)
        data.create_dataset('time/Nframe', data=Nframe)

        # Creating empty wavefunction datasets to store data:
        data.create_dataset('wavefunction/psi', (Nx, Ny, 1), maxshape=(Nx, Ny, None), dtype='complex64')

        # Stores initial state:
        data.create_dataset('initial_state/psi', data=np.fft.ifft2(psi_k))


# ---------------------------------------------------------------------------------------------------------------------
# Real time evolution
# ---------------------------------------------------------------------------------------------------------------------
for i in range(Nt):

    # Kinetic energy:
    psi_k = ne.evaluate("psi_k * exp(-0.25 * 1j * dt * (Kx ** 2 + Ky ** 2))")

    # Backward FFT:
    fft_backward(psi_k, psi)

    # Interaction term:
    psi = ne.evaluate("psi * exp(-1j * dt * c0 * (abs(psi) ** 2))")

    # Forward FFT:
    fft_forward(psi, psi_k)

    # Kinetic energy:
    psi_k = ne.evaluate("psi_k * exp(-0.25 * 1j * dt * (Kx ** 2 + Ky ** 2))")

    # Saves data
    if np.mod(i + 1, Nframe) == 0:
        with h5py.File(data_path, 'r+') as data:
            new_psi = data['wavefunction/psi']
            new_psi.resize((Nx, Ny, k + 1))
            new_psi[:, :, k] = np.fft.ifft2(psi_k)
        k += 1

    # Saves 'backup' wavefunction we can use to continue simulations if ended:
    if np.mod(i + 1, 50000) == 0:
        with h5py.File(backup_data_path, 'w') as backup:
            backup.create_dataset('time', data=t)
            backup.create_dataset('wavefunction/psi_k', shape=psi_k.shape, dtype='complex128', data=psi_k)
            backup.create_dataset('array_index', data=k)

    # Prints current time
    if np.mod(i, Nframe) == 0:
        print('t = %1.4f' % t)

    t += dt
