import numpy as np
import cupy as cp
import h5py


def rotation(wfn, Nx, Ny, alpha, beta, gamma):
    wfn_new = cp.empty((3, Nx, Ny), dtype='complex64')
    U = cp.empty((3, 3), dtype='complex64')

    # Spin-1 rotation matrix
    U[0, 0] = cp.exp(-1j * (alpha + gamma)) * (cp.cos(beta / 2)) ** 2
    U[0, 1] = -cp.exp(-1j * alpha) / cp.sqrt(2.) * cp.sin(beta)
    U[0, 2] = cp.exp(-1j * (alpha - gamma)) * (cp.sin(beta / 2)) ** 2
    U[1, 0] = cp.exp(-1j * gamma) / cp.sqrt(2.) * cp.sin(beta)
    U[1, 1] = cp.cos(beta)
    U[1, 2] = -cp.exp(1j * gamma) / cp.sqrt(2) * cp.sin(beta)
    U[2, 0] = cp.exp(1j * (alpha - gamma)) * (cp.sin(beta / 2)) ** 2
    U[2, 1] = cp.exp(1j * alpha) / cp.sqrt(2.) * cp.sin(beta)
    U[2, 2] = cp.exp(1j * (alpha + gamma)) * (cp.cos(beta / 2)) ** 2

    wfn_new[0, :, :] = U[0, 0] * wfn[0, :, :] + U[0, 1] * wfn[1, :, :] + U[0, 2] * wfn[2, :, :]
    wfn_new[1, :, :] = U[1, 0] * wfn[0, :, :] + U[1, 1] * wfn[1, :, :] + U[1, 2] * wfn[2, :, :]
    wfn_new[2, :, :] = U[2, 0] * wfn[0, :, :] + U[2, 1] * wfn[1, :, :] + U[2, 2] * wfn[2, :, :]
    return wfn_new[0, :, :], wfn_new[1, :, :], wfn_new[2, :, :]


def fourier_space(wfn_plus, wfn_0, wfn_minus, dt, Kx, Ky, q):
    wfn_plus *= cp.exp(-0.25 * 1j * dt * (Kx ** 2 + Ky ** 2 + 4 * q))
    wfn_0 *= cp.exp(-0.25 * 1j * dt * (Kx ** 2 + Ky ** 2))
    wfn_minus *= cp.exp(-0.25 * 1j * dt * (Kx ** 2 + Ky ** 2 + 4 * q))


def calc_spin_dens(wfn_plus, wfn_0, wfn_minus, time_step):
    spin_perp = cp.sqrt(2.) * (cp.conj(wfn_plus) * wfn_0 + cp.conj(wfn_0) * wfn_minus)
    spin_z = cp.abs(wfn_plus) ** 2 - cp.abs(wfn_minus) ** 2
    F = cp.sqrt(cp.abs(spin_z) ** 2 + cp.abs(spin_perp) ** 2)  # Magnitude of spin vector

    cos_term = cp.cos(c1 * F * time_step)
    sin_term = 1j * cp.sin(c1 * F * time_step) / F
    sin_term = cp.nan_to_num(sin_term)  # Corrects division by 0

    dens = cp.abs(wfn_minus) ** 2 + cp.abs(wfn_0) ** 2 + cp.abs(wfn_plus) ** 2

    return spin_perp, cp.conj(spin_perp), spin_z, cos_term, sin_term, dens


def do_fft(wfn_plus, wfn_0, wfn_minus, direction):
    if direction == 'forward':
        return cp.fft.fft2(wfn_plus), cp.fft.fft2(wfn_0), cp.fft.fft2(wfn_minus)
    elif direction == 'backward':
        return cp.fft.ifft2(wfn_plus), cp.fft.ifft2(wfn_0), cp.fft.ifft2(wfn_minus)


def interaction_flow(wfn_plus, wfn_0, wfn_minus, C, S, Fz, F_perp, F_perp_c, dt, V, p, c0, n):
    new_wfn_plus = (C * wfn_plus - (S * (Fz * wfn_plus + F_perp_c / cp.sqrt(2) * wfn_0))) * cp.exp(
        -1j * (dt * (V - p + c0 * n)))
    new_wfn_0 = (C * wfn_0 - (S / cp.sqrt(2) * (F_perp * wfn_plus + F_perp_c * wfn_minus))) * cp.exp(
        -1j * (dt * (V + c0 * n)))
    new_wfn_minus = (C * wfn_minus - (S * (F_perp / cp.sqrt(2) * wfn_0 - Fz * wfn_minus))) * cp.exp(
        -1j * (dt * (V + p + c0 * n)))

    return new_wfn_plus, new_wfn_0, new_wfn_minus


# --------------------------------------------------------------------------------------------------------------------
# Spatial and Potential parameters:
# --------------------------------------------------------------------------------------------------------------------
Nx, Ny = 1024, 1024
Mx, My = Nx // 2, Ny // 2
dx = dy = 1  # Grid spacing
dkx = np.pi / (Mx * dx)
dky = np.pi / (My * dy)  # K-space spacing
len_x = Nx * dx  # Box length
len_y = Ny * dy
x = cp.arange(-Mx, Mx) * dx
y = cp.arange(-My, My) * dy
X, Y = cp.meshgrid(x, y)  # Spatial meshgrid

kx = cp.fft.fftshift(cp.arange(-Mx, Mx) * dkx)
ky = cp.fft.fftshift(cp.arange(-My, My) * dky)
Kx, Ky = cp.meshgrid(kx, ky)  # K-space meshgrid

# Framework for wavefunction data
psi_plus = cp.empty((Nx, Ny), dtype='complex64')
psi_0 = cp.empty((Nx, Ny), dtype='complex64')
psi_minus = cp.empty((Nx, Ny), dtype='complex64')

# Controlled variables
V = 0.  # Doubly periodic box
p = 0   # Linear Zeeman
q = -0.5    # Quadratic Zeeman
c0 = 3e-5
c1 = 0.75e-5

# Time steps, number and wavefunction save variables
Nt = 10000000
Nframe = 10000  # Saves data every Nframe timesteps
dt = 1e-2  # Timestep
t = 0.
k = 0  # Array index

filename = '1024nomag'  # Name of file to save data to
data_path = '../../scratch/data/polar_phase/{}.hdf5'.format(filename)
backup_data_path = '../../scratch/data/polar_phase/backups/{}_backup.hdf5'.format(filename)

fresh_simulation = True  # Boolean that corresponds to a fresh simulation if True or a continued simulation if False

# --------------------------------------------------------------------------------------------------------------------
# Generating initial state:
# --------------------------------------------------------------------------------------------------------------------
# Loads the backup data to continue simulation and skip imaginary time evolution
if not fresh_simulation:
    previous_data = h5py.File(backup_data_path, 'r')
    psi_plus_k = cp.array(previous_data['wavefunction/psi_plus_k'])
    psi_0_k = cp.array(previous_data['wavefunction/psi_0_k'])
    psi_minus_k = cp.array(previous_data['wavefunction/psi_minus_k'])
    t = np.round(previous_data['time'][...])
    k = previous_data['array_index'][...]
    previous_data.close()

# If it is a fresh simulation, generate the initial state:
else:
    # Euler angles
    alpha = 0.
    beta = np.pi / 2 + 0.001
    gamma = 0.

    atom_num = 1.6e9 / (Nx * Ny)

    # Draw values of theta from Gaussian distribution:
    np.random.seed(9973)
    theta_k = cp.asarray(np.random.uniform(low=0, high=1, size=(Nx, Ny)) * 2 * np.pi)

    # Generate density array so that only certain modes are occupied:
    n_k = cp.zeros((Nx, Ny))
    n_k[Nx // 2, Ny // 2] = atom_num * 0.5 / (dx * dy)  # k = (0, 0)
    n_k[Nx // 2 + 1, Ny // 2] = atom_num * 0.125 / (dx * dy)  # k = (1, 0)
    n_k[Nx // 2 + 1, Ny // 2 + 1] = atom_num * 0.125 / (dx * dy)  # k = (1, 1)
    n_k[Nx // 2 - 1, Ny // 2] = atom_num * 0.125 / (dx * dy)  # k = (-1, 0)
    n_k[Nx // 2 - 1, Ny // 2 - 1] = atom_num * 0.125 / (dx * dy)  # k = (-1, -1)

    # Initial wavefunction
    Psi_k = cp.empty((3, Nx, Ny), dtype='complex64')
    Psi_k[0, :, :] = cp.zeros((Nx, Ny)) + 0j
    Psi_k[1, :, :] = cp.fft.fftshift(Nx * Ny * cp.sqrt(n_k) * cp.exp(1j * theta_k))
    Psi_k[2, :, :] = cp.zeros((Nx, Ny)) + 0j
    psi_plus_k, psi_0_k, psi_minus_k = rotation(Psi_k, Nx, Ny, alpha, beta, gamma)  # Performs rotation

    with h5py.File(data_path, 'w') as data:
        # Saving spatial data:
        data.create_dataset('grid/x', x.shape, data=cp.asnumpy(x))
        data.create_dataset('grid/y', y.shape, data=cp.asnumpy(y))

        # Saving time variables:
        data.create_dataset('time/Nt', data=Nt)
        data.create_dataset('time/dt', data=dt)
        data.create_dataset('time/Nframe', data=Nframe)

        # Creating empty wavefunction datasets to store data:
        data.create_dataset('wavefunction/psi_plus', (Nx, Ny, 1), maxshape=(Nx, Ny, None), dtype='complex64')
        data.create_dataset('wavefunction/psi_0', (Nx, Ny, 1), maxshape=(Nx, Ny, None), dtype='complex64')
        data.create_dataset('wavefunction/psi_minus', (Nx, Ny, 1), maxshape=(Nx, Ny, None), dtype='complex64')

        # Saving initial state:
        data.create_dataset('initial_state/psi_plus', data=cp.asnumpy(cp.fft.ifft2(psi_plus_k)))
        data.create_dataset('initial_state/psi_0', data=cp.asnumpy(cp.fft.ifft2(psi_0_k)))
        data.create_dataset('initial_state/psi_minus', data=cp.asnumpy(cp.fft.ifft2(psi_minus_k)))

# --------------------------------------------------------------------------------------------------------------------
# Real time evolution
# --------------------------------------------------------------------------------------------------------------------
for i in range(Nt):

    fourier_space(psi_plus_k, psi_0_k, psi_minus_k, dt, Kx, Ky, q)

    psi_plus, psi_0, psi_minus = do_fft(psi_plus_k, psi_0_k, psi_minus_k, 'backward')

    F_perp, F_perp_c, Fz, C, S, n = calc_spin_dens(psi_plus, psi_0, psi_minus, dt)

    psi_plus, psi_0, psi_minus = interaction_flow(psi_plus, psi_0, psi_minus, C, S, Fz, F_perp, F_perp_c, dt, V, p, c0,
                                                  n)

    psi_plus_k, psi_0_k, psi_minus_k = do_fft(psi_plus, psi_0, psi_minus, 'forward')

    fourier_space(psi_plus_k, psi_0_k, psi_minus_k, dt, Kx, Ky, q)

    # Saves data
    if np.mod(i + 1, Nframe) == 0:
        # Updates file with new wavefunction values:
        with h5py.File(data_path, 'r+') as data:
            new_psi_plus = data['wavefunction/psi_plus']
            new_psi_plus.resize((Nx, Ny, k + 1))
            new_psi_plus[:, :, k] = cp.asnumpy(cp.fft.ifft2(psi_plus_k))

            new_psi_0 = data['wavefunction/psi_0']
            new_psi_0.resize((Nx, Ny, k + 1))
            new_psi_0[:, :, k] = cp.asnumpy(cp.fft.ifft2(psi_0_k))

            new_psi_minus = data['wavefunction/psi_minus']
            new_psi_minus.resize((Nx, Ny, k + 1))
            new_psi_minus[:, :, k] = cp.asnumpy(cp.fft.ifft2(psi_minus_k))

        k += 1  # Increment array index

    # Saves 'backup' wavefunction we can use to continue simulations if ended:
    if np.mod(i + 1, 50000) == 0:
        with h5py.File(backup_data_path, 'w') as backup:
            backup.create_dataset('time', data=t)

            backup.create_dataset('wavefunction/psi_plus_k', shape=psi_plus_k.shape,
                                  dtype='complex64', data=cp.asnumpy(psi_plus_k))

            backup.create_dataset('wavefunction/psi_0_k', shape=psi_0_k.shape,
                                  dtype='complex64', data=cp.asnumpy(psi_0_k))

            backup.create_dataset('wavefunction/psi_minus_k', shape=psi_minus_k.shape,
                                  dtype='complex64', data=cp.asnumpy(psi_minus_k))

            backup.create_dataset('array_index', data=k)

    if np.mod(i, Nframe) == 0:
        print('t = {:2f}'.format(t))

    t += dt
