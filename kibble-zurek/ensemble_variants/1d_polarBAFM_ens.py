import cupy as cp
import h5py
import include.symplectic as sm
import sys

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
psi_plus = (cp.random.normal(0, 0.5, Nx) + 1j * cp.random.normal(0, 0.5, Nx)) / cp.sqrt(Nx)
psi_0 = cp.ones(Nx, dtype='complex64')
psi_minus = (cp.random.normal(0, 0.5, Nx) + 1j * cp.random.normal(0, 0.5, Nx)) / cp.sqrt(Nx)

psi_plus_k = cp.fft.fft(psi_plus)
psi_0_k = cp.fft.fft(psi_0)
psi_minus_k = cp.fft.fft(psi_minus)

# Controlled variables
V = 0.  # Doubly periodic box
p = 0  # Linear Zeeman
c0 = 10
c2 = -0.5
q_init = 3 * abs(c2)
q = q_init  # Quadratic Zeeman

# Time steps, number and wavefunction save variables
Nt = 200000
Nframe = 1000  # Saves data every Nframe time steps
dt = 5e-3  # Time step
t = 0.
k = 0  # Array index

filename = '1d_polar-BA-FM'  # Name of file to save data to
data_path = '../data/1d_kibble-zurek/{}.hdf5'.format(filename)
quench_rate = int(sys.argv[-2])
run_num = int(sys.argv[-1])

with h5py.File(data_path, 'a') as data:
    try:
        # Saving spatial data:
        data.create_dataset('grid/x', X.shape, data=cp.asnumpy(X))

        # Saving time variables:
        data.create_dataset('time/Nt', data=Nt)
        data.create_dataset('time/dt', data=dt)
        data.create_dataset('time/Nframe', data=Nframe)
    except (ValueError, OSError):
        print('Datasets already exist, skipping...')

    # Creating empty wavefunction datasets to store data:
    data.create_dataset('{}/run{}/wavefunction/psi_plus'.format(quench_rate, run_num), (Nx, 1), maxshape=(Nx, None),
                        dtype='complex64')
    data.create_dataset('{}/run{}/wavefunction/psi_0'.format(quench_rate, run_num), (Nx, 1), maxshape=(Nx, None),
                        dtype='complex64')
    data.create_dataset('{}/run{}/wavefunction/psi_minus'.format(quench_rate, run_num), (Nx, 1), maxshape=(Nx, None),
                        dtype='complex64')

    # Saving initial state:
    data.create_dataset('{}/run{}/initial_state/psi_plus'.format(quench_rate, run_num),
                        data=cp.asnumpy(cp.fft.ifft(psi_plus_k)))
    data.create_dataset('{}/run{}/initial_state/psi_0'.format(quench_rate, run_num),
                        data=cp.asnumpy(cp.fft.ifft(psi_0_k)))
    data.create_dataset('{}/run{}/initial_state/psi_minus'.format(quench_rate, run_num),
                        data=cp.asnumpy(cp.fft.ifft(psi_minus_k)))

# --------------------------------------------------------------------------------------------------------------------
# Real time evolution
# --------------------------------------------------------------------------------------------------------------------
for i in range(Nt):

    sm.fourier_space_1d(psi_plus_k, psi_0_k, psi_minus_k, dt, Kx, q)

    psi_plus, psi_0, psi_minus = cp.fft.ifft(psi_plus_k), cp.fft.ifft(psi_0_k), cp.fft.ifft(psi_minus_k)

    F_perp, Fz, C, S, n = sm.calc_spin_dens(psi_plus, psi_0, psi_minus, dt, c2)

    psi_plus, psi_0, psi_minus = sm.interaction_flow(psi_plus, psi_0, psi_minus, C, S, Fz, F_perp, dt, V, p, c0, n)

    psi_plus_k, psi_0_k, psi_minus_k = cp.fft.fft(psi_plus), cp.fft.fft(psi_0), cp.fft.fft(psi_minus)

    sm.fourier_space_1d(psi_plus_k, psi_0_k, psi_minus_k, dt, Kx, q)

    # Decrease q linearly until we meet threshold
    if q > -q_init:
        q = q_init * (1 - t / quench_rate)

    # Saves data
    if cp.mod(i + 1, Nframe) == 0:
        # Updates file with new wavefunction values:
        with h5py.File(data_path, 'r+') as data:
            new_psi_plus = data['{}/run{}/wavefunction/psi_plus'.format(quench_rate, run_num)]
            new_psi_plus.resize((Nx, k + 1))
            new_psi_plus[:, k] = cp.asnumpy(cp.fft.ifft(psi_plus_k))

            new_psi_0 = data['{}/run{}/wavefunction/psi_0'.format(quench_rate, run_num)]
            new_psi_0.resize((Nx, k + 1))
            new_psi_0[:, k] = cp.asnumpy(cp.fft.ifft(psi_0_k))

            new_psi_minus = data['{}/run{}/wavefunction/psi_minus'.format(quench_rate, run_num)]
            new_psi_minus.resize((Nx, k + 1))
            new_psi_minus[:, k] = cp.asnumpy(cp.fft.ifft(psi_minus_k))

        k += 1  # Increment array index

    if cp.mod(i, Nframe) == 0:
        print('t = {:2f}'.format(t))

    t += dt
