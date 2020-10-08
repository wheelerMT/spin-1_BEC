import numpy as np
import cupy as cp
import numexpr as ne
from numpy import heaviside as heav
import h5py


def get_phase(N_vort, pos, Nx, Ny, X, Y, len_x, len_y):
    # Phase initialisation
    theta_k = np.empty((N_vort, Nx, Ny))
    theta_tot = np.empty((Nx, Ny))
    pi = np.pi

    for k in range(N_vort // 2):
        y_m, y_p = pos[k], pos[N_vort + k]  # y-positions
        x_m, x_p = pos[N_vort // 2 + k], pos[3 * N_vort // 2 + k]  # x-positions

        # Scaling positional arguments
        Y_minus = 2 * np.pi * (Y - y_m) / len_y
        X_minus = 2 * np.pi * (X - x_m) / len_x
        Y_plus = 2 * np.pi * (Y - y_p) / len_y
        X_plus = 2 * np.pi * (X - x_p) / len_x
        x_plus = 2 * np.pi * x_p / len_x
        x_minus = 2 * np.pi * x_m / len_x

        heav_xp = heav(X_plus, 1.)
        heav_xm = heav(X_minus, 1.)

        for nn in np.arange(-5, 5):
            theta_k[k, :, :] += ne.evaluate("arctan(tanh((Y_minus + 2 * pi * nn) / 2) * tan((X_minus - pi) / 2)) "
                                            "- arctan(tanh((Y_plus + 2 * pi * nn) / 2) * tan((X_plus - pi) / 2)) "
                                            "+ pi * (heav_xp - heav_xm)")

        theta_k[k, :, :] -= ne.evaluate("(2 * pi * Y / len_y) * (x_plus - x_minus) / (2 * pi)")
        theta_tot += theta_k[k, :, :]
    return theta_tot


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
    wfn_plus *= cp.exp(-0.25 * 1j * dt * (Kx ** 2 + Ky ** 2 + 2 * q))
    wfn_0 *= cp.exp(-0.25 * 1j * dt * (Kx ** 2 + Ky ** 2))
    wfn_minus *= cp.exp(-0.25 * 1j * dt * (Kx ** 2 + Ky ** 2 + 2 * q))


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
q = -0.01    # Quadratic Zeeman
c0 = 3e-5
c1 = 0.75e-5

# Time steps, number and wavefunction save variables
Nt = 5000000
Nframe = 10000  # Saves data every Nframe timesteps
dt = 1e-2  # Timestep
t = 0.
k = 0  # Array index

filename = '1024nomag_imprinted'  # Name of file to save data to
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
    beta = 0.
    gamma = 0.

    n_0 = 1.6e9 / (Nx * Ny)  # Background density

    # Generate phase fields:
    N_vort_sqv = 60
    N_vort_hqv = 60

    # Generate initial SQV positions for both phase fields
    sqv_pos = [pos for pos in np.random.uniform(-len_x // 2, len_x // 2, size=N_vort_sqv * 2)]

    # Generate extra positions for HQVs in plus component:
    plus_vortices = [pos for pos in sqv_pos]
    minus_vortices = [pos for pos in sqv_pos]

    for i in range(N_vort_hqv // 2):
        # Generates random positions for HQVs, keeping the SQV positions
        plus_vortices.insert(len(sqv_pos) // 4, np.random.uniform(-len_x // 2, len_x // 2))
        plus_vortices.insert(len(sqv_pos) // 2, np.random.uniform(-len_x // 2, len_x // 2))
        plus_vortices.insert(3 * len(sqv_pos) // 4, np.random.uniform(-len_x // 2, len_x // 2))
        plus_vortices.insert(len(sqv_pos), np.random.uniform(-len_x // 2, len_x // 2))

        minus_vortices.insert(len(sqv_pos) // 4, np.random.uniform(-len_x // 2, len_x // 2))
        minus_vortices.insert(len(sqv_pos) // 2, np.random.uniform(-len_x // 2, len_x // 2))
        minus_vortices.insert(3 * len(sqv_pos) // 4, np.random.uniform(-len_x // 2, len_x // 2))
        minus_vortices.insert(len(sqv_pos), np.random.uniform(-len_x // 2, len_x // 2))

    theta_plus = get_phase(N_vort_sqv + N_vort_sqv, plus_vortices, Nx, Ny, cp.asnumpy(X), cp.asnumpy(Y), len_x, len_y)
    theta_minus = get_phase(N_vort_sqv + N_vort_sqv, minus_vortices, Nx, Ny, cp.asnumpy(X), cp.asnumpy(Y), len_x, len_y)

    # Initial wavefunction
    Psi_k = cp.empty((3, Nx, Ny), dtype='complex64')
    Psi_k[0, :, :] = cp.sqrt(n_0) * cp.exp(1j * cp.asarray(theta_plus)) / cp.sqrt(2)
    Psi_k[1, :, :] = cp.zeros((Nx, Ny)) + 0j + 1e-10
    Psi_k[2, :, :] = cp.sqrt(n_0) * cp.exp(1j * cp.asarray(theta_minus)) / cp.sqrt(2)
    psi_plus, psi_0, psi_minus = rotation(Psi_k, Nx, Ny, alpha, beta, gamma)  # Performs rotation

    N_plus = dx * dy * cp.sum(cp.abs(psi_plus) ** 2)
    N_0 = dx * dy * cp.sum(cp.abs(psi_0) ** 2)
    N_minus = dx * dy * cp.sum(cp.abs(psi_minus) ** 2)

    psi_plus_k = cp.fft.fft2(psi_plus)
    psi_0_k = cp.fft.fft2(psi_0)
    psi_minus_k = cp.fft.fft2(psi_minus)

# --------------------------------------------------------------------------------------------------------------------
# Imaginary time evolution
# --------------------------------------------------------------------------------------------------------------------
if fresh_simulation is True:
    for i in range(200):
        fourier_space(psi_plus_k, psi_0_k, psi_minus_k, -1j * dt, Kx, Ky, q)

        psi_plus, psi_0, psi_minus = do_fft(psi_plus_k, psi_0_k, psi_minus_k, 'backward')

        F_perp, F_perp_c, Fz, C, S, n = calc_spin_dens(psi_plus, psi_0, psi_minus, -1j * dt)

        psi_plus, psi_0, psi_minus = interaction_flow(psi_plus, psi_0, psi_minus, C, S, Fz, F_perp, F_perp_c, -1j * dt, V,
                                                      p, c0,
                                                      n)

        psi_plus_k, psi_0_k, psi_minus_k = do_fft(psi_plus, psi_0, psi_minus, 'forward')

        fourier_space(psi_plus_k, psi_0_k, psi_minus_k, -1j * dt, Kx, Ky, q)

        N_plus_new = dx * dy * cp.sum(cp.abs(cp.fft.ifft2(psi_plus_k)) ** 2)
        N_0_new = dx * dy * cp.sum(cp.abs(cp.fft.ifft2(psi_0_k)) ** 2)
        N_minus_new = dx * dy * cp.sum(cp.abs(cp.fft.ifft2(psi_minus_k)) ** 2)

        psi_plus_k = cp.fft.fft2(cp.sqrt(N_plus) * cp.fft.ifft2(psi_plus_k) / cp.sqrt(N_plus_new))
        psi_0_k = cp.fft.fft2(cp.sqrt(N_0) * cp.fft.ifft2(psi_0_k) / cp.sqrt(N_0_new))
        psi_minus_k = cp.fft.fft2(cp.sqrt(N_minus) * cp.fft.ifft2(psi_minus_k) / cp.sqrt(N_minus_new))

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
