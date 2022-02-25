import numpy as np
import h5py
import pyfftw
import include.symplectic_cpu as sm
import sys

# --------------------------------------------------------------------------------------------------------------------
# Spatial and Potential parameters:
# --------------------------------------------------------------------------------------------------------------------
Nx = 4096
Mx = Nx // 2
dx = 0.25  # Grid spacing
dkx = np.pi / (Mx * dx)
len_x = Nx * dx  # Box length
X = np.arange(-Mx, Mx) * dx

Kx = np.arange(-Mx, Mx) * dkx
Kx = np.fft.fftshift(Kx)

# Framework for wavefunction FFTs
psi_plus = pyfftw.empty_aligned(Nx, dtype='complex64')
psi_0 = pyfftw.empty_aligned(Nx, dtype='complex64')
psi_minus = pyfftw.empty_aligned(Nx, dtype='complex64')
psi_plus_k = pyfftw.empty_aligned(Nx, dtype='complex64')
psi_0_k = pyfftw.empty_aligned(Nx, dtype='complex64')
psi_minus_k = pyfftw.empty_aligned(Nx, dtype='complex64')
fft_plus = pyfftw.builders.fft(psi_plus)
fft_0 = pyfftw.builders.fft(psi_0)
fft_minus = pyfftw.builders.fft(psi_minus)
ifft_plus = pyfftw.builders.ifft(psi_plus)
ifft_0 = pyfftw.builders.ifft(psi_0)
ifft_minus = pyfftw.builders.ifft(psi_minus)

# Set wavefunction
n_0 = 1.  # Background density
psi_plus[:] = ((np.random.normal(0, 0.5, Nx) + 1j * np.random.normal(0, 0.5, Nx)) / np.sqrt(Nx)).astype('complex64')
psi_0[:] = np.sqrt(n_0) * np.ones(Nx, dtype='complex64')
psi_minus[:] = ((np.random.normal(0, 0.5, Nx) + 1j * np.random.normal(0, 0.5, Nx)) / np.sqrt(Nx)).astype('complex64')

psi_plus_k[:] = fft_plus(psi_plus)
psi_0_k[:] = fft_plus(psi_0)
psi_minus_k[:] = fft_plus(psi_minus)

# Controlled variables
V = 0.  # Doubly periodic box
p = 0  # Linear Zeeman
c0 = 10
c2 = -0.5

# Time steps, number and wavefunction save variables
tau_q = int(sys.argv[-2])
run_num = int(sys.argv[-1])

dt = 1e-3  # Time step
Nframe = 1000  # Number of frames of data
t = -0.5 * tau_q  # Choose this so Q_init = 2.5
Q_init = 2 - t / tau_q
Q = Q_init
Nt = int(2 * Q_init * tau_q / dt)
N_steps = Nt // Nframe  # Saves data every N_steps
k = 0  # Array index

filename = f'1d_polar-BA-FM_{run_num}'  # Name of file to save data to
data_path = f'../scratch/data/spin-1/kibble-zurek/ensembles/tau_q={tau_q}/{filename}.hdf5'

with h5py.File(data_path, 'a') as data:
    try:
        # Saving spatial data:
        data.create_dataset('grid/x', X.shape, data=X)

        # Saving time variables:
        data.create_dataset('time/dt', data=dt)
        data.create_dataset('time/Nframe', data=Nframe)

    except (ValueError, OSError, RuntimeError):
        print('Datasets already exist, skipping...')

    try:
        # Save variable time data
        data.create_dataset('{}/time/Nt'.format(tau_q), data=Nt)
        data.create_dataset('{}/time/N_steps'.format(tau_q), data=N_steps)

    except (ValueError, OSError, RuntimeError):
        print('Datasets already exist, skipping...')

    # Creating empty wavefunction datasets to store data:
    data.create_dataset('wavefunction/psi_plus'.format(tau_q, run_num), (Nx, 1), maxshape=(Nx, None),
                        dtype='complex64')
    data.create_dataset('wavefunction/psi_0'.format(tau_q, run_num), (Nx, 1), maxshape=(Nx, None),
                        dtype='complex64')
    data.create_dataset('wavefunction/psi_minus'.format(tau_q, run_num), (Nx, 1), maxshape=(Nx, None),
                        dtype='complex64')

    # Saving initial state:
    data.create_dataset('initial_state/psi_plus'.format(tau_q, run_num),
                        data=psi_plus)
    data.create_dataset('initial_state/psi_0'.format(tau_q, run_num),
                        data=psi_0)
    data.create_dataset('initial_state/psi_minus'.format(tau_q, run_num),
                        data=psi_minus)

# --------------------------------------------------------------------------------------------------------------------
# Real time evolution
# --------------------------------------------------------------------------------------------------------------------
for i in range(Nt):

    sm.fourier_space_KZ_1d(psi_plus_k, psi_0_k, psi_minus_k, dt, Kx, Q, c2, n_0, tau_q, sign=-1)

    psi_plus, psi_0, psi_minus = ifft_plus(psi_plus_k), ifft_0(psi_0_k), ifft_minus(psi_minus_k)

    F_perp, Fz, C, S, n = sm.calc_spin_dens(psi_plus, psi_0, psi_minus, dt, c2)

    psi_plus, psi_0, psi_minus = sm.interaction_flow(psi_plus, psi_0, psi_minus, C, S, Fz, F_perp, dt, V, p, c0, n)

    psi_plus_k, psi_0_k, psi_minus_k = fft_plus(psi_plus), fft_0(psi_0), fft_minus(psi_minus)

    sm.fourier_space_KZ_1d(psi_plus_k, psi_0_k, psi_minus_k, dt, Kx, Q, c2, n_0, tau_q, sign=-1)

    # Decrease q linearly until we meet threshold
    if Q > -Q_init:
        Q = -t / tau_q

    # Saves data
    if np.mod(i + 1, N_steps) == 0:
        # Updates file with new wavefunction values:
        with h5py.File(data_path, 'r+') as data:
            new_psi_plus = data['wavefunction/psi_plus'.format(tau_q, run_num)]
            new_psi_plus.resize((Nx, k + 1))
            new_psi_plus[:, k] = ifft_plus(psi_plus_k)

            new_psi_0 = data['wavefunction/psi_0'.format(tau_q, run_num)]
            new_psi_0.resize((Nx, k + 1))
            new_psi_0[:, k] = ifft_0(psi_0_k)

            new_psi_minus = data['wavefunction/psi_minus'.format(tau_q, run_num)]
            new_psi_minus.resize((Nx, k + 1))
            new_psi_minus[:, k] = ifft_minus(psi_minus_k)

        k += 1  # Increment array index

    if np.mod(i, N_steps) == 0:
        print('t = {:2f}'.format(t))

    t += dt
    i += 1
