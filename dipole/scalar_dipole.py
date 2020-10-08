import numpy as np
import multiprocessing as mp
import pyfftw
from numpy import pi, exp, sqrt, sin, cos, conj, arctan, tanh, tan
from numpy import heaviside as heav
import matplotlib.pyplot as plt
import h5py


# ---------Spatial and potential parameters--------------
Mx = My = 64
Nx = Ny = 128  # Number of grid pts
dx = dy = 1 / 2  # Grid spacing
dkx = np.pi / (Mx * dx)
dky = np.pi / (My * dy)  # K-space spacing
len_x = Nx * dx     # Box length
len_y = Ny * dy
x = np.arange(-Mx, Mx) * dx
y = np.arange(-My, My) * dy
X, Y = np.meshgrid(x, y)  # Spatial meshgrid

kx = np.fft.fftshift(np.arange(-Mx, Mx) * dkx)
ky = np.fft.fftshift(np.arange(-My, My) * dky)
Kx, Ky = np.meshgrid(kx, ky)  # K-space meshgrid

# Initialising FFTs
cpu_count = mp.cpu_count()
wfn_data = pyfftw.empty_aligned((Nx, Ny), dtype='complex128')
fft_forward = pyfftw.FFTW(wfn_data, wfn_data, axes=(0, 1), threads=cpu_count)
fft_backward = pyfftw.FFTW(wfn_data, wfn_data, direction='FFTW_BACKWARD', axes=(0, 1), threads=cpu_count)

# Framework for wavefunction data
psi_k = pyfftw.empty_aligned((Nx, Ny), dtype='complex128')

# Controlled variables
V = 0.  # Doubly periodic box
c0 = 2
k = 0
N_vort = 2
pos = [-5.5, -10, 5.5, -10]
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

psi = exp(1j * theta_tot)
pyfftw.byte_align(psi)
theta_init = np.angle(psi)
N = dx * dy * np.linalg.norm(psi) ** 2

# Time steps, number and wavefunction save variables
Nt = 20000
Nframe = 200
dt = 5e-3
t = 0.

# Creating file for saving data:
data = h5py.File('../data/scalar_data.hdf5', 'w')
psi_save = data.create_dataset('wavefunction/psi', (Nx, Ny, Nt/Nframe), dtype='complex128')

for i in range(1000):
    fft_forward(psi, psi_k)
    psi_k *= exp(-0.25 * dt * (Kx ** 2 + Ky ** 2)) / (Nx * Ny)
    fft_backward(psi_k, psi)

    psi *= (Nx * Ny)

    psi *= exp(-dt * c0 * (abs(psi)**2))

    fft_forward(psi, psi_k)
    psi_k *= exp(-0.25 * dt * (Kx ** 2 + Ky ** 2)) / (Nx * Ny)
    fft_backward(psi_k, psi)

    psi *= (Nx * Ny)

    psi *= sqrt(N) / sqrt(dx * dy * np.linalg.norm(psi) ** 2)

    psi *= exp(1j * theta_init) / exp(1j * np.angle(psi))
    if np.mod(i, Nframe) == 0:
        print('it = %1.4f' % t)

    t += dt

t = 0

for i in range(Nt):
    fft_forward(psi, psi_k)
    psi_k *= exp(-0.25 * 1j * dt * (Kx ** 2 + Ky ** 2)) / (Nx * Ny)
    fft_backward(psi_k, psi)

    psi *= (Nx * Ny)

    psi *= exp(-1j * dt * c0 * (abs(psi) ** 2))

    fft_forward(psi, psi_k)
    psi_k *= exp(-0.25 * 1j * dt * (Kx ** 2 + Ky ** 2)) / (Nx * Ny)
    fft_backward(psi_k, psi)

    psi *= (Nx * Ny)

    if np.mod(i, Nframe) == 0:
        print('t = %1.4f' % t)
        psi_save[:, :, k] = psi[:, :]
        k += 1

    t += dt

data.close()


