import numpy as np
from numpy import pi
import h5py
import cupy as cp


# --------------------------------------------------------------------------------------------------------------------
# Controlled variables:
# --------------------------------------------------------------------------------------------------------------------
Nx, Ny = 1024, 1024
Mx, My = Nx // 2, Ny // 2  # Number of grid pts
dx = dy = 0.5  # Grid spacing
dkx = pi / (Mx * dx)
dky = pi / (My * dy)  # K-space spacing
len_x = Nx * dx  # Box length
len_y = Ny * dy
x = cp.arange(-Mx, Mx) * dx
y = cp.arange(-My, My) * dy
X, Y = cp.meshgrid(x, y)  # Spatial meshgrid

# k-space arrays and meshgrid:
kx = cp.fft.fftshift(cp.arange(-Mx, Mx) * dkx)
ky = cp.fft.fftshift(cp.arange(-My, My) * dky)
Kx, Ky = cp.meshgrid(kx, ky)  # K-space meshgrid

# Initialising FFTs
psi = cp.empty((Nx, Ny), dtype='complex64')

# Controlled variables
V = 0.  # Doubly periodic box
c0 = 3e-5

# Time steps, number and wavefunction save variables
Nt = 10000000
Nframe = 10000   # Save data every Nframe number of timesteps
dt = 1e-2  # Imaginary time timestep
t = 0.
k = 0   # Array index

filename = 'scalar'    # Name of file to save data to
data_path = '../scratch/data/scalar/{}.hdf5'.format(filename)
backup_data_path = '../scratch/data/scalar/{}_backup.hdf5'.format(filename)

fresh_simulation = True  # Boolean that corresponds to a fresh simulation if True or a continued simulation if False

# --------------------------------------------------------------------------------------------------------------------
# Generating initial state:
# --------------------------------------------------------------------------------------------------------------------
# If it is a continued simulation, load the previous data and continue to evolution:
if not fresh_simulation:
    previous_data = h5py.File(backup_data_path, 'r')
    psi_k = cp.array(previous_data['wavefunction/psi_k'])
    t = np.round(previous_data['time'][...])
    k = previous_data['array_index'][...]
    previous_data.close()

# If it is a fresh simulation, generate the initial state:
else:
    atom_num = 4e8 / (Nx * Ny)

    # Draw values of theta from Gaussian distribution:
    np.random.seed(9971)
    theta_k = cp.asarray(np.random.uniform(low=0, high=1, size=(Nx, Ny)) * 2 * np.pi)

    # Generate density array so that only certain modes are occupied:
    n_k = cp.zeros((Nx, Ny))
    n_k[Nx // 2, Ny // 2] = atom_num * 0.5 / (dx * dy)    # k = (0, 0)
    n_k[Nx // 2 + 1, Ny // 2] = atom_num * 0.125 / (dx * dy)  # k = (1, 0)
    n_k[Nx // 2 + 1, Ny // 2 + 1] = atom_num * 0.125 / (dx * dy)  # k = (1, 1)
    n_k[Nx // 2 - 1, Ny // 2] = atom_num * 0.125 / (dx * dy)  # k = (-1, 0)
    n_k[Nx // 2 - 1, Ny // 2 - 1] = atom_num * 0.125 / (dx * dy)  # k = (-1, -1)

    # Construct wavefunction in Fourier space:
    psi_k = cp.fft.ifftshift(Nx * Ny * cp.sqrt(n_k) * cp.exp(1j * theta_k))

    # Creating file to save to:
    with h5py.File(data_path, 'w') as data:
        # Saving spatial data:
        data.create_dataset('grid/x', x.shape, data=cp.asnumpy(x))
        data.create_dataset('grid/y', y.shape, data=cp.asnumpy(y))

        # Saving time variables:
        data.create_dataset('time/Nt', data=Nt)
        data.create_dataset('time/dt', data=dt)
        data.create_dataset('time/Nframe', data=Nframe)

        # Creating empty wavefunction datasets to store data:
        data.create_dataset('wavefunction/psi', (Nx, Ny, 1), maxshape=(Nx, Ny, None), dtype='complex64')

        # Stores initial state:
        data.create_dataset('initial_state/psi', data=cp.asnumpy(cp.fft.ifft2(psi_k)))


# ---------------------------------------------------------------------------------------------------------------------
# Real time evolution
# ---------------------------------------------------------------------------------------------------------------------
for i in range(Nt):

    # Kinetic energy:
    psi_k *= cp.exp(-0.25 * 1j * dt * (Kx ** 2 + Ky ** 2))

    # Backward FFT:
    psi = cp.fft.ifft2(psi_k)

    # Interaction term:
    psi *= cp.exp(-1j * dt * (c0 * cp.abs(psi) ** 2))

    # Forward FFT:
    psi_k = cp.fft.fft2(psi)

    # Kinetic energy:
    psi_k *= cp.exp(-0.25 * 1j * dt * (Kx ** 2 + Ky ** 2))

    # Saves data
    if np.mod(i + 1, Nframe) == 0:
        with h5py.File(data_path, 'r+') as data:
            new_psi = data['wavefunction/psi']
            new_psi.resize((Nx, Ny, k + 1))
            new_psi[:, :, k] = cp.asnumpy(cp.fft.ifft2(psi_k))
        k += 1

    # Saves 'backup' wavefunction we can use to continue simulations if ended:
    if np.mod(i + 1, 50000) == 0:
        with h5py.File(backup_data_path, 'w') as backup:
            backup.create_dataset('time', data=t)
            backup.create_dataset('wavefunction/psi_k', shape=psi_k.shape, dtype='complex64', data=cp.asnumpy(psi_k))
            backup.create_dataset('array_index', data=k)

    # Prints current time
    if np.mod(i, Nframe) == 0:
        print('t = %1.4f' % t)

    t += dt
