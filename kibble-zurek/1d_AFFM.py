import cupy as cp
import h5py
import include.symplectic as sm
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------------------------------------
# Spatial and Potential parameters:
# --------------------------------------------------------------------------------------------------------------------
Nx = 1024
Mx = Nx // 2
dx = 0.5  # Grid spacing
dkx = cp.pi / (Mx * dx)
len_x = Nx * dx  # Box length
X = cp.arange(-Mx, Mx) * dx

Kx = cp.arange(-Mx, Mx) * dkx
Kx = cp.fft.fftshift(Kx)

# Controlled variables
V = 0.  # Doubly periodic box
p_init = 0.
p_final = 1
p = p_init  # Linear Zeeman
q = -0.5  # Quadratic Zeeman
quench_time = 500  # Time when p=1 (dimensionless units)
c0 = 10
c2 = 0.5

# Framework for wavefunction data
psi_plus = cp.ones(Nx, dtype='complex64') * cp.sqrt((1 + p / (c2 * 1)) / 2) \
           * cp.exp(1j * cp.random.uniform(0, 2 * cp.pi, Nx))
psi_0 = (cp.random.normal(0, 0.5, Nx) + 1j * cp.random.normal(0, 0.5, Nx)) / cp.sqrt(Nx)
psi_minus = cp.ones(Nx, dtype='complex64') * cp.sqrt((1 - p / (c2 * 1)) / 2) \
            * cp.exp(1j * cp.random.uniform(0, 2 * cp.pi, Nx))

# Add noise to k=0-10 modes
psi_plus_k = cp.fft.fftshift(cp.fft.fft2(psi_plus))
psi_0_k = cp.fft.fft2(psi_0)
psi_minus_k = cp.fft.fftshift(cp.fft.fft2(psi_minus))

for k_x in range(11):
    psi_plus_k[k_x] += cp.sqrt(Nx) * cp.sqrt(1 / 2) * cp.exp(1j * cp.random.uniform(0, 2 * cp.pi))
    psi_minus_k[k_x] += cp.sqrt(Nx) * cp.sqrt(1 / 2) * cp.exp(1j * cp.random.uniform(0, 2 * cp.pi))

psi_plus_k = cp.fft.ifftshift(psi_plus_k)
psi_minus_k = cp.fft.ifftshift(psi_minus_k)

atom_num_plus = dx * cp.sum(cp.abs(psi_plus) ** 2)
atom_num_0 = dx * cp.sum(cp.abs(psi_0) ** 2)
atom_num_minus = dx * cp.sum(cp.abs(psi_minus) ** 2)
atom_num = atom_num_plus + atom_num_0 + atom_num_minus

target_mag = cp.sum(abs(psi_plus) ** 2 - abs(psi_minus) ** 2) / 1024

# Time steps, number and wavefunction save variables
Nt = 200000
Nframe = 1000  # Saves data every Nframe time steps
dt = 5e-3  # Time step
t = 0.
k = 0  # Array index

filename = '1d_AF-FM'  # Name of file to save data to
data_path = '../data/1d_kibble-zurek/{}.hdf5'.format(filename)

with h5py.File(data_path, 'w') as data:
    # Saving spatial data:
    data.create_dataset('grid/x', X.shape, data=cp.asnumpy(X))

    # Saving time variables:
    data.create_dataset('time/Nt', data=Nt)
    data.create_dataset('time/dt', data=dt)
    data.create_dataset('time/Nframe', data=Nframe)
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

    sm.fourier_space_1d(psi_plus_k, psi_0_k, psi_minus_k, dt, Kx, q)

    psi_plus, psi_0, psi_minus = cp.fft.ifft(psi_plus_k), cp.fft.ifft(psi_0_k), cp.fft.ifft(psi_minus_k)

    F_perp, Fz, C, S, n = sm.calc_spin_dens(psi_plus, psi_0, psi_minus, dt, c2)

    psi_plus, psi_0, psi_minus = sm.interaction_flow(psi_plus, psi_0, psi_minus, C, S, Fz, F_perp, dt, V, p, c0, n)

    psi_plus_k, psi_0_k, psi_minus_k = cp.fft.fft(psi_plus), cp.fft.fft(psi_0), cp.fft.fft(psi_minus)

    sm.fourier_space_1d(psi_plus_k, psi_0_k, psi_minus_k, dt, Kx, q)

    # Increase p linearly until we meet threshold
    if p < p_final:
        p = p_final * t / quench_time

    # Saves data
    if cp.mod(i + 1, Nframe) == 0:
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

    if cp.mod(i, Nframe) == 0:
        # print(q)
        print('t = {:2f}'.format(t))

    t += dt
