import numpy as np
import cupy as cp
import h5py
import include.symplectic as sm

# --------------------------------------------------------------------------------------------------------------------
# Spatial and Potential parameters:
# --------------------------------------------------------------------------------------------------------------------
Nx, Ny = 512, 512
Mx, My = Nx // 2, Ny // 2
dx = dy = 1  # Grid spacing
dkx = np.pi / (Mx * dx)
dky = np.pi / (My * dy)  # K-space spacing
len_x = Nx * dx  # Box length
len_y = Ny * dy
x = cp.arange(-Mx, Mx) * dx
y = cp.arange(-My, My) * dy
X, Y = cp.meshgrid(x, y)  # Spatial meshgrid

kx = cp.arange(-Mx, Mx) * dkx
ky = cp.arange(-My, My) * dky
Kx, Ky = cp.meshgrid(kx, ky)  # K-space meshgrid
Kx, Ky = cp.fft.fftshift(Kx), cp.fft.fftshift(Ky)

# Controlled variables
V = 0.  # Doubly periodic box
p_init = 0.
p_final = 1
p = p_init  # Linear Zeeman
q = -0.5  # Quadratic Zeeman
quench_time = 1000  # Time when p=1 (dimensionless units)
c0 = 10
c2 = 0.5

# Framework for wavefunction data
psi_plus = cp.ones((Nx, Ny), dtype='complex64') * cp.sqrt((1 + p / (c2 * 1)) / 2)
psi_0 = (cp.random.normal(0, 0.5, (Nx, Ny)) + 1j * cp.random.normal(0, 0.5, (Nx, Ny))) / cp.sqrt(Nx * Ny)
psi_minus = cp.ones((Nx, Ny), dtype='complex64') * cp.sqrt((1 - p / (c2 * 1)) / 2)

# Add noise to k=0-10 modes
psi_plus_k = cp.fft.fftshift(cp.fft.fft2(psi_plus))
psi_0_k = cp.fft.fft2(psi_0)
psi_minus_k = cp.fft.fftshift(cp.fft.fft2(psi_minus))

for k_x in range(Nx):
    for k_y in range(Ny):
        if int(cp.sqrt(k_x ** 2 + k_y ** 2)) <= 10:
            psi_plus_k[k_x, k_y] += cp.sqrt(Nx * Ny) * cp.sqrt(1 / 2) * cp.exp(1j * cp.random.uniform(0, 2 * cp.pi))
            psi_minus_k[k_x, k_y] += cp.sqrt(Nx * Ny) * cp.sqrt(1 / 2) * cp.exp(1j * cp.random.uniform(0, 2 * cp.pi))

psi_plus_k = cp.fft.ifftshift(psi_plus_k)
psi_minus_k = cp.fft.ifftshift(psi_minus_k)

atom_num_plus = dx * dy * cp.sum(cp.abs(psi_plus) ** 2)
atom_num_0 = dx * dy * cp.sum(cp.abs(psi_0) ** 2)
atom_num_minus = dx * dy * cp.sum(cp.abs(psi_minus) ** 2)
atom_num = atom_num_plus + atom_num_0 + atom_num_minus

# Time steps, number and wavefunction save variables
Nt = 200000 * 2
Nframe = 1000  # Saves data every Nframe time steps
dt = 5e-3  # Time step
t = 0.
k = 0  # Array index

filename = '2d_AF-FM'  # Name of file to save data to
data_path = '../scratch/data/spin-1/kibble-zurek/{}.hdf5'.format(filename)

fresh_simulation = True  # Boolean that corresponds to a fresh simulation if True or a continued simulation if False

# --------------------------------------------------------------------------------------------------------------------
# Generating initial state:
# --------------------------------------------------------------------------------------------------------------------
# Loads the backup data to continue simulation and skip imaginary time evolution
# if not fresh_simulation:
# previous_data = h5py.File(backup_data_path, 'r')
# psi_plus_k = cp.array(previous_data['wavefunction/psi_plus_k'])
# psi_0_k = cp.array(previous_data['wavefunction/psi_0_k'])
# psi_minus_k = cp.array(previous_data['wavefunction/psi_minus_k'])
# t = np.round(previous_data['time'][...])
# k = previous_data['array_index'][...]
# previous_data.close()

# If it is a fresh simulation, generate the initial state:
# else:
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

    sm.fourier_space(psi_plus_k, psi_0_k, psi_minus_k, dt, Kx, Ky, q)

    psi_plus, psi_0, psi_minus = cp.fft.ifft2(psi_plus_k), cp.fft.ifft2(psi_0_k), cp.fft.ifft2(psi_minus_k)

    F_perp, Fz, C, S, n = sm.calc_spin_dens(psi_plus, psi_0, psi_minus, dt, c2)

    psi_plus, psi_0, psi_minus = sm.interaction_flow(psi_plus, psi_0, psi_minus, C, S, Fz, F_perp, dt, V, p, c0, n)

    psi_plus_k, psi_0_k, psi_minus_k = cp.fft.fft2(psi_plus), cp.fft.fft2(psi_0), cp.fft.fft2(psi_minus)

    sm.fourier_space(psi_plus_k, psi_0_k, psi_minus_k, dt, Kx, Ky, q)

    # psi_plus = cp.fft.ifft2(psi_plus_k)
    # psi_0 = cp.fft.ifft2(psi_0_k)
    # psi_minus = cp.fft.ifft2(psi_minus_k)
    #
    # atom_num_new_plus = dx * dy * cp.sum(cp.abs(psi_plus) ** 2)
    # atom_num_new_0 = dx * dy * cp.sum(cp.abs(psi_0) ** 2)
    # atom_num_new_minus = dx * dy * cp.sum(cp.abs(psi_minus) ** 2)
    #
    # atom_num_new = atom_num_new_minus + atom_num_new_0 + atom_num_new_plus
    #
    # psi_plus *= cp.sqrt(atom_num / atom_num_new)
    # psi_0 *= cp.sqrt(atom_num / atom_num_new)
    # psi_minus *= cp.sqrt(atom_num / atom_num_new)
    #
    # psi_plus_k = cp.fft.fft2(psi_plus)
    # psi_0_k = cp.fft.fft2(psi_0)
    # psi_minus_k = cp.fft.fft2(psi_minus)

    # Increase p linearly until we meet threshold
    if p < p_final:
        p = p_final * t / quench_time

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

    if np.mod(i, Nframe) == 0:
        print('t = {:2f}'.format(t))

    t += dt
