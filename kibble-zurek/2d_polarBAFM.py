import cupy as cp
import h5py
import include.symplectic as sm

# --------------------------------------------------------------------------------------------------------------------
# Spatial and Potential parameters:
# --------------------------------------------------------------------------------------------------------------------
Nx, Ny = 1024, 1024
Mx, My = Nx // 2, Ny // 2
dx, dy = 0.5, 0.5  # Grid spacing
dkx, dky = cp.pi / (Mx * dx), cp.pi / (My * dy)
len_x, len_y = Nx * dx, Ny * dy  # Box length
x, y = cp.arange(-Mx, Mx) * dx, cp.arange(-My, My) * dy
X, Y = cp.meshgrid(x, y, indexing='ij')
kx, ky = cp.arange(-Mx, Mx) * dkx, cp.arange(-My, My) * dky
Kx, Ky = cp.meshgrid(kx, ky, indexing='ij')
Kx, Ky = cp.fft.fftshift(Kx), cp.fft.fftshift(Ky)

# Framework for wavefunction data
n_0 = 1.
psi_plus = (cp.random.normal(0, 0.5, (Nx, Ny)) + 1j * cp.random.normal(0, 0.5, (Nx, Ny))) / cp.sqrt(Nx * Ny)
psi_0 = cp.ones((Nx, Ny), dtype='complex64') + (cp.random.normal(0, 0.5, (Nx, Ny)) +
                                                1j * cp.random.normal(0, 0.5, (Nx, Ny))) / cp.sqrt(Nx * Ny)
psi_minus = (cp.random.normal(0, 0.5, (Nx, Ny)) + 1j * cp.random.normal(0, 0.5, (Nx, Ny))) / cp.sqrt(Nx * Ny)

psi_plus_k = cp.fft.fft2(psi_plus)
psi_0_k = cp.fft.fft2(psi_0)
psi_minus_k = cp.fft.fft2(psi_minus)

# Controlled variables
V = 0.  # Doubly periodic box
p = 0  # Linear Zeeman
tau_q = 5000  # Time when q=-q_init (dimensionless units)
c0 = 10
c2 = -0.5

# Time steps, number and wavefunction save variables
dt = 1e-3  # Time step
Nframe = 2000  # Number of frames of data
Q_init = 2.5
t = -Q_init * tau_q  # Choose this so Q_init = 2.5
Q = Q_init
Nt = int(1.5 * Q_init * tau_q / dt)  # 1.5 is to extend beyond when Q = -Q_init
N_steps = Nt // Nframe  # Saves data every N_steps
k = 0  # Array index

filename = '2d_polar-BA-FM_{}'.format(tau_q)  # Name of file to save data to
data_path = '../scratch/data/spin-1/kibble-zurek/{}.hdf5'.format(filename)

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

    sm.fourier_space_KZ_2d(psi_plus_k, psi_0_k, psi_minus_k, dt, Kx, Ky, Q, c2, n_0, tau_q, sign=-1)

    psi_plus, psi_0, psi_minus = cp.fft.ifft2(psi_plus_k), cp.fft.ifft2(psi_0_k), cp.fft.ifft2(psi_minus_k)

    F_perp, Fz, C, S, n = sm.calc_spin_dens(psi_plus, psi_0, psi_minus, dt, c2)

    psi_plus, psi_0, psi_minus = sm.interaction_flow(psi_plus, psi_0, psi_minus, C, S, Fz, F_perp, dt, V, p, c0, n)

    psi_plus_k, psi_0_k, psi_minus_k = cp.fft.fft2(psi_plus), cp.fft.fft2(psi_0), cp.fft.fft2(psi_minus)

    sm.fourier_space_KZ_2d(psi_plus_k, psi_0_k, psi_minus_k, dt, Kx, Ky, Q, c2, n_0, tau_q, sign=-1)

    # Decrease q linearly until we meet threshold
    if Q > -Q_init:
        Q = - t / tau_q

    # Saves data
    if cp.mod(i + 1, Nframe) == 0:
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

    if cp.mod(i, Nframe) == 0:
        print(Q)
        print('t = {:2f}'.format(t))

    t += dt