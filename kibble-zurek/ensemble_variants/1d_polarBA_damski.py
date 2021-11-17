import cupy as cp
import h5py
import sys
# sys.path.insert(0, 'C:/dev/uni/spin-1/') # location of src
import include.symplectic as sm

# --------------------------------------------------------------------------------------------------------------------
# Spatial and Potential parameters:
# --------------------------------------------------------------------------------------------------------------------
Nx = 2048
Mx = Nx // 2
dx = 78 / Nx  # Grid spacing
dkx = cp.pi / (Mx * dx)
len_x = Nx * dx  # Box length
X = cp.arange(-Mx, Mx) * dx
Kx = cp.arange(-Mx, Mx) * dkx
Kx = cp.fft.fftshift(Kx)

# Framework for wavefunction data
n_0 = 1 / len_x  # Background density
psi_plus = (cp.random.normal(0, 1e-4, Nx) + 1j * cp.random.normal(0, 1e-4, Nx)) / cp.sqrt(Nx)
psi_0 = cp.sqrt(n_0) + ((cp.random.normal(0, 1e-4, Nx) + 1j * cp.random.normal(0, 1e-4, Nx)) / cp.sqrt(Nx))
psi_minus = (cp.random.normal(0, 1e-4, Nx) + 1j * cp.random.normal(0, 1e-4, Nx)) / cp.sqrt(Nx)

psi_plus_k = cp.fft.fft(psi_plus)
psi_0_k = cp.fft.fft(psi_0)
psi_minus_k = cp.fft.fft(psi_minus)

# Controlled variables
V = 0.  # Doubly periodic box
p = 0  # Linear Zeeman
c0 = 1.4e5
c2 = -630
q = 2.5 * abs(c2) * n  # Initial quadratic Zeeman
quench_time = int(sys.argv[-1])
run_num = 1  # Change this to int(sys.argv[-2]) for more runs

# Calculate number of time steps needed
dt = 5e-5  # Time step
Nframe = 1000  # Number of frames of data
t = -0.5 * quench_time  # Choose this so Q_init = 2.5
Q_init = 2 - t / quench_time
Q = Q_init
Nt = 2 * Q_init * quench_time / dt
N_steps = Nt // Nframe  # Saves data every N_steps

# Initialise variables
k = 0  # Array index

filename = '1d_polar-BA_damski'  # Name of file to save data to
data_path = '../../data/1d_kibble-zurek/{}.hdf5'.format(filename)
with h5py.File(data_path, 'a') as data:
    try:
        # Saving spatial data:
        data.create_dataset('grid/x', X.shape, data=cp.asnumpy(X))

        # Saving time variables:
        data.create_dataset('time/dt', data=dt)
    except (ValueError, OSError):
        print('Datasets already exist, skipping...')

    # Saving time variables
    data.create_dataset('{}/time/Nt'.format(quench_time), data=Nt)
    data.create_dataset('{}/time/Nframe'.format(quench_time), data=Nframe)

    # Creating empty wavefunction datasets to store data:
    data.create_dataset('{}/run{}/wavefunction/psi_plus'.format(quench_time, run_num), (Nx, 1), maxshape=(Nx, None),
                        dtype='complex64')
    data.create_dataset('{}/run{}/wavefunction/psi_0'.format(quench_time, run_num), (Nx, 1), maxshape=(Nx, None),
                        dtype='complex64')
    data.create_dataset('{}/run{}/wavefunction/psi_minus'.format(quench_time, run_num), (Nx, 1), maxshape=(Nx, None),
                        dtype='complex64')

    # Saving initial state:
    data.create_dataset('{}/run{}/initial_state/psi_plus'.format(quench_time, run_num),
                        data=cp.asnumpy(cp.fft.ifft(psi_plus_k)))
    data.create_dataset('{}/run{}/initial_state/psi_0'.format(quench_time, run_num),
                        data=cp.asnumpy(cp.fft.ifft(psi_0_k)))
    data.create_dataset('{}/run{}/initial_state/psi_minus'.format(quench_time, run_num),
                        data=cp.asnumpy(cp.fft.ifft(psi_minus_k)))

# --------------------------------------------------------------------------------------------------------------------
# Real time evolution
# --------------------------------------------------------------------------------------------------------------------
i = 0
while Q > 0:

    sm.fourier_space_KZ_1d(psi_plus_k, psi_0_k, psi_minus_k, dt, Kx, Q, c2, n_0, quench_time)

    psi_plus, psi_0, psi_minus = cp.fft.ifft(psi_plus_k), cp.fft.ifft(psi_0_k), cp.fft.ifft(psi_minus_k)

    F_perp, Fz, C, S, n = sm.calc_spin_dens(psi_plus, psi_0, psi_minus, dt, c2)

    psi_plus, psi_0, psi_minus = sm.interaction_flow(psi_plus, psi_0, psi_minus, C, S, Fz, F_perp, dt, V, p, c0, n)

    psi_plus_k, psi_0_k, psi_minus_k = cp.fft.fft(psi_plus), cp.fft.fft(psi_0), cp.fft.fft(psi_minus)

    sm.fourier_space_KZ_1d(psi_plus_k, psi_0_k, psi_minus_k, dt, Kx, Q, c2, n_0, quench_time)

    if sm.transverse_mag(cp.fft.ifft(psi_plus_k), cp.fft.ifft(psi_0_k), cp.fft.ifft(psi_minus_k), dx) >= 0.01:
        with h5py.File(data_path, 'r+') as data:
            data.create_dataset('{}/run{}/t_hat'.format(quench_time, run_num), data=t)
        print('t_hat for quench time {} = {}'.format(quench_time, t))
        exit()
    # Decrease q linearly until we meet threshold
    q = abs(c2) * n_0 * (2 - t / quench_time)

    # Saves data
    if cp.mod(i + 1, Nframe) == 0:
        # Updates file with new wavefunction values:
        with h5py.File(data_path, 'r+') as data:
            new_psi_plus = data['{}/run{}/wavefunction/psi_plus'.format(quench_time, run_num)]
            new_psi_plus.resize((Nx, k + 1))
            new_psi_plus[:, k] = cp.asnumpy(cp.fft.ifft(psi_plus_k))

            new_psi_0 = data['{}/run{}/wavefunction/psi_0'.format(quench_time, run_num)]
            new_psi_0.resize((Nx, k + 1))
            new_psi_0[:, k] = cp.asnumpy(cp.fft.ifft(psi_0_k))

            new_psi_minus = data['{}/run{}/wavefunction/psi_minus'.format(quench_time, run_num)]
            new_psi_minus.resize((Nx, k + 1))
            new_psi_minus[:, k] = cp.asnumpy(cp.fft.ifft(psi_minus_k))

        k += 1  # Increment array index

    if cp.mod(i, Nframe) == 0:
        print('t = {:2f}'.format(t))

    t += dt
    i += 1
