import numpy as np
import multiprocessing as mp
import pyfftw
from numpy import pi, exp, sqrt, sin, cos, conj, arctan, tanh, tan
from numpy import heaviside as heav
from include import helper
import h5py


# ---------Spatial and potential parameters--------------
Mx = My = 64
Nx = Ny = 128  # Number of grid pts
dx = dy = 1 / 2  # Grid spacing
dkx = pi / (Mx * dx)
dky = pi / (My * dy)  # K-space spacing
len_x = Nx * dx     # Box length
len_y = Ny * dy
x = np.arange(-Mx, Mx) * dx
y = np.arange(-My, My) * dy
X, Y = np.meshgrid(x, y)  # Spatial meshgrid

data = h5py.File('../data/splitting_dipole_data.hdf5', 'a')
data.create_dataset('grid/x', x.shape, data=x)
data.create_dataset('grid/y', y.shape, data=y)

kx = np.fft.fftshift(np.arange(-Mx, Mx) * dkx)
ky = np.fft.fftshift(np.arange(-My, My) * dky)
Kx, Ky = np.meshgrid(kx, ky)  # K-space meshgrid

# Initialising FFTs
cpu_count = mp.cpu_count()
wfn_data = pyfftw.empty_aligned((Nx, Ny), dtype='complex128')
fft_forward = pyfftw.FFTW(wfn_data, wfn_data, axes=(0, 1), threads=cpu_count)
fft_backward = pyfftw.FFTW(wfn_data, wfn_data, direction='FFTW_BACKWARD', axes=(0, 1), threads=cpu_count)

# Framework for wavefunction data
psi_plus_k = pyfftw.empty_aligned((Nx, Ny), dtype='complex128')
psi_0_k = pyfftw.empty_aligned((Nx, Ny), dtype='complex128')
psi_minus_k = pyfftw.empty_aligned((Nx, Ny), dtype='complex128')

# Controlled variables
V = 0.  # Doubly periodic box
p = q = 0.
c0 = 2
c1 = 0.5  # Effective 3-component BEC
k = 0  # Array index

# ------------------------------ Generating SQV's -------------------------
# Euler angles
alpha = 0.
beta = pi / 4
gamma = 0.

N_vort = 2  # Number of vortices

pos = [-10, 0, 10, 0]
theta_k = np.empty((N_vort, Nx, Ny))
theta_tot = np.empty((Nx, Ny))

for k in range(N_vort // 2):
    # Scaling positional arguments
    Y_minus = 2 * pi * (Y - pos[k]) / len_y
    X_minus = 2 * pi * (X - pos[N_vort // 2 + k]) / len_x
    Y_plus = 2 * pi * (Y - pos[N_vort + k]) / len_y
    X_plus = 2 * pi * (X - pos[3 * N_vort // 2 + k]) / len_x
    x_plus = 2 * pi * pos[3 * N_vort // 2 + k] / len_x
    x_minus = 2 * pi * pos[N_vort // 2 + k] / len_x

    for nn in np.arange(-5, 5):
        theta_k[k, :, :] += arctan(
            tanh((Y_minus + 2 * pi * nn) / 2) * tan((X_minus - pi) / 2)) \
                            - arctan(tanh((Y_plus + 2 * pi * nn) / 2) * tan((X_plus - pi) / 2)) \
                            + pi * (heav(X_plus, 1.) - heav(X_minus, 1.))

    theta_k[k, :, :] -= (2 * pi * Y / len_y) * (x_plus - x_minus) / (2 * pi)
    theta_tot += theta_k[k, :, :]


# Initial wavefunction
Psi = np.empty((3, Nx, Ny), dtype='complex128')
Psi[0, :, :] = np.zeros((Nx, Ny)) + 0j
Psi[1, :, :] = np.ones((Nx, Ny), dtype='complex128') * exp(1j * theta_tot)
Psi[2, :, :] = np.zeros((Nx, Ny)) + 0j
psi_plus, psi_0, psi_minus = helper.rotation(Psi, Nx, Ny, alpha, beta, gamma)  # Performs rotation to wavefunction

# Aligning wavefunction to potentially speed up FFTs
pyfftw.byte_align(psi_plus)
pyfftw.byte_align(psi_0)
pyfftw.byte_align(psi_minus)
# ------------------------------------------------------------------------

# Normalisation constants
N_plus = dx * dy * np.linalg.norm(psi_plus) ** 2
N_0 = dx * dy * np.linalg.norm(psi_0) ** 2
N_minus = dx * dy * np.linalg.norm(psi_minus) ** 2

# Time steps, number and wavefunction save variables
Nt = 80000
Nframe = 200
dt = 5e-3
t = 0.

# Saving time variables:
data.create_dataset('time/Nt', data=Nt)
data.create_dataset('time/dt', data=dt)
data.create_dataset('time/Nframe', data=Nframe)

# Setting up variables to be sequentially saved:
psi_plus_save = data.create_dataset('wavefunction/psi_plus', (Nx, Ny, Nt/Nframe), dtype='complex128')
psi_0_save = data.create_dataset('wavefunction/psi_0', (Nx, Ny, Nt/Nframe), dtype='complex128')
psi_minus_save = data.create_dataset('wavefunction/psi_minus', (Nx, Ny, Nt/Nframe), dtype='complex128')

for i in range(Nt):
    # Spin vector terms:
    F_perp = sqrt(2.) * (conj(psi_plus) * psi_0 + conj(psi_0) * psi_minus)
    Fz = abs(psi_plus) ** 2 - abs(psi_minus) ** 2
    F = sqrt(abs(Fz) ** 2 + abs(F_perp) ** 2)  # Magnitude of spin vector

    # Total density
    n = abs(psi_minus) ** 2 + abs(psi_0) ** 2 + abs(psi_plus) ** 2

    # Sin and cosine terms for solution
    C = cos(c1 * F * (-1j * dt))
    if F.min() == 0:
        S = np.zeros((Nx, Ny), dtype='complex128')  # Ensures no division by zero
    else:
        S = 1j * sin(c1 * F * (-1j * dt)) / F

    # Forward FFTs
    fft_forward(psi_plus, psi_plus_k)
    fft_forward(psi_0, psi_0_k)
    fft_forward(psi_minus, psi_minus_k)

    # Computing kinetic energy + quadratic Zeeman
    psi_plus_k *= exp(-0.25 * dt * (Kx ** 2 + Ky ** 2 + 2 * q)) / (Nx * Ny)
    psi_0_k *= exp(-0.25 * dt * (Kx ** 2 + Ky ** 2)) / (Nx * Ny)
    psi_minus_k *= exp(-0.25 * dt * (Kx ** 2 + Ky ** 2 + 2 * q)) / (Nx * Ny)

    # Inverse FFTs
    fft_backward(psi_plus_k, psi_plus)
    fft_backward(psi_0_k, psi_0)
    fft_backward(psi_minus_k, psi_minus)

    # Rescaling
    psi_plus *= (Nx * Ny)
    psi_0 *= (Nx * Ny)
    psi_minus *= (Nx * Ny)

    # Trap, linear Zeeman & interaction flow
    psi_plus = ((C - S * Fz) * psi_plus - 1. / sqrt(2.) * S * conj(F_perp) * psi_0) * exp(-dt * (V - p + c0 * n))
    psi_0 = (-1. / sqrt(2.) * S * F_perp * psi_plus + C * psi_0 - 1. / sqrt(2.) * S * conj(F_perp) * psi_minus) \
            * exp(-dt * (V + c0 * n))
    psi_minus = (-1. / sqrt(2.) * S * F_perp * psi_0 + (C + S * Fz) * psi_minus) * exp(-dt * (V + p + c0 * n))

    # Forward FFTs
    fft_forward(psi_plus, psi_plus_k)
    fft_forward(psi_0, psi_0_k)
    fft_forward(psi_minus, psi_minus_k)

    # Computing kinetic energy + quadratic Zeeman
    psi_plus_k *= exp(-0.25 * dt * (Kx ** 2 + Ky ** 2 + 2 * q)) / (Nx * Ny)
    psi_0_k *= exp(-0.25 * dt * (Kx ** 2 + Ky ** 2)) / (Nx * Ny)
    psi_minus_k *= exp(-0.25 * dt * (Kx ** 2 + Ky ** 2 + 2 * q)) / (Nx * Ny)

    # Inverse FFTs
    fft_backward(psi_plus_k, psi_plus)
    fft_backward(psi_0_k, psi_0)
    fft_backward(psi_minus_k, psi_minus)

    # Rescaling
    psi_plus *= (Nx * Ny)
    psi_0 *= (Nx * Ny)
    psi_minus *= (Nx * Ny)

    # Renormalizing wavefunction
    psi_plus *= sqrt(N_plus) / sqrt(dx * dy * np.linalg.norm(psi_plus) ** 2)
    psi_0 *= sqrt(N_0) / sqrt(dx * dy * np.linalg.norm(psi_0) ** 2)
    psi_minus *= sqrt(N_minus) / sqrt(dx * dy * np.linalg.norm(psi_minus) ** 2)

    # Prints current time and saves data to an array
    if np.mod(i, Nframe) == 0:
        print('it = %1.4f' % t)

        psi_plus_save[:, :, k] = psi_plus[:, :]
        psi_0_save[:, :, k] = psi_0[:, :]
        psi_minus_save[:, :, k] = psi_minus[:, :]

        k += 1
    t += dt

data.close()
