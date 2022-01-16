import cupy as cp
import h5py
import include.symplectic as sm

# --------------------------------------------------------------------------------------------------------------------
# Spatial and Potential parameters:
# --------------------------------------------------------------------------------------------------------------------
Nx = 8192 * 2
Mx = Nx // 2
dx = 0.125  # Grid spacing
dkx = cp.pi / (Mx * dx)
len_x = Nx * dx  # Box length
X = cp.arange(-Mx, Mx) * dx

Kx = cp.arange(-Mx, Mx) * dkx
Kx = cp.fft.fftshift(Kx)

# Framework for wavefunction data
n_0 = 1.
psi_plus = (cp.random.normal(0, 0.5, Nx) + 1j * cp.random.normal(0, 0.5, Nx)) / cp.sqrt(Nx)
psi_0 = cp.ones(Nx, dtype='complex64') + (cp.random.normal(0, 0.5, Nx) + 1j * cp.random.normal(0, 0.5, Nx)) / cp.sqrt(Nx)
psi_minus = (cp.random.normal(0, 0.5, Nx) + 1j * cp.random.normal(0, 0.5, Nx)) / cp.sqrt(Nx)

psi_plus_k = cp.fft.fft(psi_plus)
psi_0_k = cp.fft.fft(psi_0)
psi_minus_k = cp.fft.fft(psi_minus)

# Controlled variables
V = 0.  # Doubly periodic box
p = 0  # Linear Zeeman
tau_q = 2500  # Time when q=-q_init (dimensionless units)
c0 = 10
c2 = -0.5

# Time steps, number and wavefunction save variables
dt = 1e-3  # Time step
Nframe = 1000  # Number of frames of data
Q_init = 2.5
t = -Q_init * tau_q  # Choose this so Q_init = 2.5
Q = Q_init
Nt = int(1.5 * Q_init * tau_q / dt)  # 1.5 is to extend beyond when Q = -Q_init
N_steps = Nt // Nframe  # Saves data every N_steps
k = 0  # Array index

filename = '1d_polar-BA-FM_{}'.format(tau_q)  # Name of file to save data to
data_path = '../scratch/data/spin-1/kibble-zurek/{}.hdf5'.format(filename)

with h5py.File(data_path, 'w') as data:
    # Saving spatial data:
    data.create_dataset('grid/x', X.shape, data=cp.asnumpy(X))

    # Saving time variables:
    data.create_dataset('time/Nt', data=Nt)
    data.create_dataset('time/dt', data=dt)
    data.create_dataset('time/Nframe', data=Nframe)
    data.create_dataset('time/N_steps', data=N_steps)
    data.create_dataset('time/t', (1, 1), maxshape=(None, 1), dtype='float64')

    # Creating empty wavefunction datasets to store data:
    data.create_dataset('wavefunction/psi_plus', (Nx, 1), maxshape=(Nx, None), dtype='complex64')
    data.create_dataset('wavefunction/psi_0', (Nx, 1), maxshape=(Nx, None), dtype='complex64')
    data.create_dataset('wavefunction/psi_minus', (Nx, 1), maxshape=(Nx, None), dtype='complex64')

    # Saving initial state:
    data.create_dataset('initial_state/psi_plus', data=cp.asnumpy(cp.fft.ifft(psi_plus_k)))
    data.create_dataset('initial_state/psi_0', data=cp.asnumpy(cp.fft.ifft(psi_0_k)))
    data.create_dataset('initial_state/psi_minus', data=cp.asnumpy(cp.fft.ifft(psi_minus_k)))

# --------------------------------------------------------------------------------------------------------------------
# Real time evolution
# --------------------------------------------------------------------------------------------------------------------
for i in range(Nt):

    sm.fourier_space_KZ_1d(psi_plus_k, psi_0_k, psi_minus_k, dt, Kx, Q, c2, n_0, tau_q, sign=-1)

    psi_plus, psi_0, psi_minus = cp.fft.ifft(psi_plus_k), cp.fft.ifft(psi_0_k), cp.fft.ifft(psi_minus_k)

    F_perp, Fz, C, S, n = sm.calc_spin_dens(psi_plus, psi_0, psi_minus, dt, c2)

    psi_plus, psi_0, psi_minus = sm.interaction_flow(psi_plus, psi_0, psi_minus, C, S, Fz, F_perp, dt, V, p, c0, n)

    psi_plus_k, psi_0_k, psi_minus_k = cp.fft.fft(psi_plus), cp.fft.fft(psi_0), cp.fft.fft(psi_minus)

    sm.fourier_space_KZ_1d(psi_plus_k, psi_0_k, psi_minus_k, dt, Kx, Q, c2, n_0, tau_q, sign=-1)

    # Decrease q linearly until we meet threshold
    if Q > -Q_init:
        Q = -t / tau_q

    # Saves data
    if cp.mod(i + 1, N_steps) == 0:
        # Updates file with new wavefunction values:
        with h5py.File(data_path, 'r+') as data:
            new_psi_plus = data['wavefunction/psi_plus']
            new_psi_plus.resize((Nx, k + 1))
            new_psi_plus[:, k] = cp.asnumpy(cp.fft.ifft(psi_plus_k))

            new_psi_0 = data['wavefunction/psi_0']
            new_psi_0.resize((Nx, k + 1))
            new_psi_0[:, k] = cp.asnumpy(cp.fft.ifft(psi_0_k))

            new_psi_minus = data['wavefunction/psi_minus']
            new_psi_minus.resize((Nx, k + 1))
            new_psi_minus[:, k] = cp.asnumpy(cp.fft.ifft(psi_minus_k))

            time_array = data['time/t']
            time_array.resize((k + 1, 1))
            time_array[k, 0] = t

        k += 1  # Increment array index

    if cp.mod(i, N_steps) == 0:
        print(Q)
        print('t = {:2f}'.format(t))

    t += dt
