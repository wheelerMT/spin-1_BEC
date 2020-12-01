import h5py
import numpy as np
import cupy as cp
from include import symplectic as sm

"""Generates a pair of half-quantum vortices in opposite spinor components of either same or opposite circulation."""


# --------------------------------------------------------------------------------------------------------------------
# Spatial and Potential parameters:
# --------------------------------------------------------------------------------------------------------------------
Nx, Ny = 256, 256  # Number of grid pts
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

# Controlled variables
V_width = 5  # Number of pts for thickness of potential barrier
V = np.where(X < Mx - V_width, 0, 1e20) + np.where(X > -Mx + V_width, 0, 1e20) + np.where(Y < My - V_width, 0, 1e20) \
    + np.where(Y > -My + V_width, 0, 1e20)  # Square box potential

p = 0.
q = -0.01
c0 = 3e-5
c1 = 0.75e-5
k = 0

# Time steps, number and wavefunction save variables
Nt = 4000000
Nframe = 2000
dt = 1e-2
t = 0.

# ----------------------------------------------------------------------------------------------------------------------
# Generating initial state
# ----------------------------------------------------------------------------------------------------------------------
# Euler angles
alpha = 0.
beta = 0.
gamma = 0.

theta_plus = cp.arctan2(Y - 15.5, X)
theta_minus = -cp.arctan2(Y + 15.5, X)

n_0 = 1.6e9 / (1024 ** 2)  # Background density
cvals = np.linspace(0, 1600, 25)

# Initial wavefunction
Psi = cp.empty((3, Nx, Ny), dtype='complex128')
Psi[0, :, :] = cp.sqrt(n_0 / 2.) * cp.ones((Nx, Ny), dtype='complex128') * cp.exp(1j * theta_plus)
Psi[1, :, :] = cp.zeros((Nx, Ny)) + 0j
Psi[2, :, :] = cp.sqrt(n_0 / 2.) * cp.ones((Nx, Ny), dtype='complex128') * cp.exp(1j * theta_minus)
psi_plus, psi_0, psi_minus = sm.rotation(Psi, Nx, Ny, alpha, beta, gamma)  # Performs rotation to wavefunction

# Finding argument of each component:
theta_plus_fix = np.angle(psi_plus)
theta_minus_fix = np.angle(psi_minus)

psi_plus_k = cp.fft.fft2(psi_plus)
psi_0_k = cp.fft.fft2(psi_0)
psi_minus_k = cp.fft.fft2(psi_minus)

# Normalisation constants
N_plus = dx * dy * cp.linalg.norm(psi_plus) ** 2
N_minus = dx * dy * cp.linalg.norm(psi_minus) ** 2

# ----------------------------------------------------------------------------------------------------------------------
# Generating data file
# ----------------------------------------------------------------------------------------------------------------------
filename = 'HQV_oppComp'  # Name of file to save data to
data_path = '../data/dipole/{}.hdf5'.format(filename)

# --------------------------------------------------------------------------------------------------------------------
# Imaginary time evolution
# --------------------------------------------------------------------------------------------------------------------
for i in range(3000):
    sm.fourier_space(psi_plus_k, psi_0_k, psi_minus_k, -1j * dt, Kx, Ky, q)

    psi_plus, psi_0, psi_minus = cp.fft.ifft2(psi_plus_k), cp.fft.ifft2(psi_0_k), cp.fft.ifft2(psi_minus_k)

    F_perp, Fz, C, S, n = sm.calc_spin_dens(psi_plus, psi_0, psi_minus, -1j * dt, c1)

    psi_plus, psi_0, psi_minus = sm.interaction_flow(psi_plus, psi_0, psi_minus, C, S, Fz, F_perp, -1j * dt, V, p, c0,
                                                     n)

    psi_plus_k, psi_0_k, psi_minus_k = cp.fft.fft2(psi_plus), cp.fft.fft2(psi_0), cp.fft.fft2(psi_minus)

    sm.fourier_space(psi_plus_k, psi_0_k, psi_minus_k, -1j * dt, Kx, Ky, q)

    N_plus_new = dx * dy * cp.sum(cp.abs(cp.fft.ifft2(psi_plus_k)) ** 2)
    N_minus_new = dx * dy * cp.sum(cp.abs(cp.fft.ifft2(psi_minus_k)) ** 2)

    psi_plus_k = cp.fft.fft2(cp.sqrt(N_plus) * cp.fft.ifft2(psi_plus_k) / cp.sqrt(N_plus_new))
    psi_minus_k = cp.fft.fft2(cp.sqrt(N_minus) * cp.fft.ifft2(psi_minus_k) / cp.sqrt(N_minus_new))

    # Fixing the phase:
    psi_plus = cp.fft.ifft2(psi_plus_k)
    psi_minus = cp.fft.ifft2(psi_minus_k)
    psi_plus *= cp.exp(1j * theta_plus_fix) / cp.exp(1j * np.angle(psi_plus))
    psi_minus *= cp.exp(1j * theta_minus_fix) / cp.exp(1j * np.angle(psi_minus))
    psi_plus_k = cp.fft.fft2(psi_plus)
    psi_minus_k = cp.fft.fft2(psi_minus)

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

    sm.fourier_space(psi_plus_k, psi_0_k, psi_minus_k, dt, Kx, Ky, q)

    psi_plus, psi_0, psi_minus = cp.fft.ifft2(psi_plus_k), cp.fft.ifft2(psi_0_k), cp.fft.ifft2(psi_minus_k)

    F_perp, Fz, C, S, n = sm.calc_spin_dens(psi_plus, psi_0, psi_minus, dt, c1)

    psi_plus, psi_0, psi_minus = sm.interaction_flow(psi_plus, psi_0, psi_minus, C, S, Fz, F_perp, dt, V, p, c0, n)

    psi_plus_k, psi_0_k, psi_minus_k = cp.fft.fft2(psi_plus), cp.fft.fft2(psi_0), cp.fft.fft2(psi_minus)

    sm.fourier_space(psi_plus_k, psi_0_k, psi_minus_k, dt, Kx, Ky, q)

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

    if np.mod(i, Nframe * 2) == 0:
        print('t = {:2f}'.format(t))

    t += dt
