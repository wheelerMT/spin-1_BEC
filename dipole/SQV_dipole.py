import numpy as np
import pyfftw
from numpy import pi, exp, sqrt, conj, arctan, tanh, tan
from numpy import heaviside as heav
from include import helper
import h5py
import numexpr as ne


# --------------------------------------------------------------------------------------------------------------------
# Spatial and Potential parameters:
# --------------------------------------------------------------------------------------------------------------------
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

kx = np.fft.fftshift(np.arange(-Mx, Mx) * dkx)
ky = np.fft.fftshift(np.arange(-My, My) * dky)
Kx, Ky = np.meshgrid(kx, ky)  # K-space meshgrid

# Initialising FFTs
wfn_data = pyfftw.empty_aligned((Nx, Ny), dtype='complex128')
fft_forward = pyfftw.FFTW(wfn_data, wfn_data, axes=(0, 1), threads=2)
fft_backward = pyfftw.FFTW(wfn_data, wfn_data, direction='FFTW_BACKWARD', axes=(0, 1), threads=2)

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

# Time steps, number and wavefunction save variables
Nt = 20000
Nframe = 100
dt = 1e-2
t = 0.

# ----------------------------------------------------------------------------------------------------------------------
# Generating initial state
# ----------------------------------------------------------------------------------------------------------------------

# Euler angles
alpha = 0.
beta = pi / 4
gamma = 0.

N_vort = 2  # Number of vortices

pos = [-5.5, -10, 5.5, -10]  # Position of dipoles
theta_k = np.empty((N_vort, Nx, Ny))
theta_tot = np.empty((Nx, Ny))

# Generates phase profile with imprinted vortices
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
Psi[1, :, :] = np.ones((Nx, Ny)) * exp(1j * theta_tot)
Psi[2, :, :] = np.zeros((Nx, Ny)) + 0j
psi_plus, psi_0, psi_minus = helper.rotation(Psi, Nx, Ny, alpha, beta, gamma)  # Performs rotation to wavefunction

# Normalisation constants
N_plus = dx * dy * np.linalg.norm(psi_plus) ** 2
N_0 = dx * dy * np.linalg.norm(psi_0) ** 2
N_minus = dx * dy * np.linalg.norm(psi_minus) ** 2

# Finding argument of each component:
theta_plus = np.angle(psi_plus)
theta_0 = np.angle(psi_0)
theta_minus = np.angle(psi_minus)

# Aligning wavefunction to potentially speed up FFTs
pyfftw.byte_align(psi_plus)
pyfftw.byte_align(psi_0)
pyfftw.byte_align(psi_minus)

# ----------------------------------------------------------------------------------------------------------------------
# Generating data file
# ----------------------------------------------------------------------------------------------------------------------
filename = 'SQV_dipole'   # Name of file to save data to
data_path = '../data/dipole/{}_data.hdf5'.format(filename)

with h5py.File(data_path, 'w') as data:
    # Saving spatial data:
    data.create_dataset('grid/x', x.shape, data=x)
    data.create_dataset('grid/y', y.shape, data=y)

    # Saving time variables:
    data.create_dataset('time/Nt', data=Nt)
    data.create_dataset('time/dt', data=dt)
    data.create_dataset('time/Nframe', data=Nframe)

    # Creating empty wavefunction datasets to store data:
    data.create_dataset('wavefunction/psi_plus', (Nx, Ny, 1), maxshape=(Nx, Ny, None), dtype='complex64')
    data.create_dataset('wavefunction/psi_0', (Nx, Ny, 1), maxshape=(Nx, Ny, None), dtype='complex64')
    data.create_dataset('wavefunction/psi_minus', (Nx, Ny, 1), maxshape=(Nx, Ny, None), dtype='complex64')


# ----------------------------------------------------------------------------------------------------------------------
# Imaginary time evolution
# ----------------------------------------------------------------------------------------------------------------------

for i in range(1000):

    # Forward FFTs
    fft_forward(psi_plus, psi_plus_k)
    fft_forward(psi_0, psi_0_k)
    fft_forward(psi_minus, psi_minus_k)

    # Computing kinetic energy + quadratic Zeeman
    psi_plus_k *= exp(-0.25 * dt * (Kx ** 2 + Ky ** 2 + 4 * q))
    psi_0_k *= exp(-0.25 * dt * (Kx ** 2 + Ky ** 2))
    psi_minus_k *= exp(-0.25 * dt * (Kx ** 2 + Ky ** 2 + 4 * q))

    # Inverse FFTs
    fft_backward(psi_plus_k, psi_plus)
    fft_backward(psi_0_k, psi_0)
    fft_backward(psi_minus_k, psi_minus)

    # Spin vector terms:
    F_perp = sqrt(2.) * (conj(psi_plus) * psi_0 + conj(psi_0) * psi_minus)
    Fz = abs(psi_plus) ** 2 - abs(psi_minus) ** 2
    F = sqrt(abs(Fz) ** 2 + abs(F_perp) ** 2)  # Magnitude of spin vector

    # Total density
    n = abs(psi_minus) ** 2 + abs(psi_0) ** 2 + abs(psi_plus) ** 2

    # Sin and cosine terms for solution
    C = ne.evaluate("cos(-1j * c1 * F * dt)")
    S = ne.evaluate("1j * sin(-1j * c1 * F * dt) / F")
    S = np.nan_to_num(S)  # Corrects division by 0

    psi_plus_old = psi_plus
    psi_0_old = psi_0
    psi_minus_old = psi_minus

    # Trap, linear Zeeman & interaction flow
    psi_plus = ne.evaluate("C * psi_plus_old - (S * (Fz * psi_plus_old + conj(F_perp) / sqrt(2) * psi_0_old))")
    psi_0 = ne.evaluate("C * psi_0_old - (S / sqrt(2) * (F_perp * psi_plus_old + conj(F_perp) * psi_minus_old))")
    psi_minus = ne.evaluate("C * psi_minus_old - (S * (F_perp / sqrt(2) * psi_0_old - Fz * psi_minus_old))")

    psi_plus *= ne.evaluate("exp(-dt * (V - p + c0 * n))")
    psi_0 *= ne.evaluate("exp(-dt * (V + c0 * n))")
    psi_minus *= ne.evaluate("exp(-dt * (V + p + c0 * n))")

    # Forward FFTs
    fft_forward(psi_plus, psi_plus_k)
    fft_forward(psi_0, psi_0_k)
    fft_forward(psi_minus, psi_minus_k)

    # Computing kinetic energy + quadratic Zeeman
    psi_plus_k *= exp(-0.25 * dt * (Kx ** 2 + Ky ** 2 + 2 * q))
    psi_0_k *= exp(-0.25 * dt * (Kx ** 2 + Ky ** 2))
    psi_minus_k *= exp(-0.25 * dt * (Kx ** 2 + Ky ** 2 + 2 * q))

    # Inverse FFTs
    fft_backward(psi_plus_k, psi_plus)
    fft_backward(psi_0_k, psi_0)
    fft_backward(psi_minus_k, psi_minus)

    # Renormalizing wavefunction
    psi_plus *= sqrt(N_plus) / sqrt(dx * dy * np.linalg.norm(psi_plus) ** 2)
    psi_0 *= sqrt(N_0) / sqrt(dx * dy * np.linalg.norm(psi_0) ** 2)
    psi_minus *= sqrt(N_minus) / sqrt(dx * dy * np.linalg.norm(psi_minus) ** 2)

    # Fixing the phase:
    psi_plus *= exp(1j * theta_plus) / exp(1j * np.angle(psi_plus))
    psi_0 *= exp(1j * theta_0) / exp(1j * np.angle(psi_0))
    psi_minus *= exp(1j * theta_minus) / exp(1j * np.angle(psi_minus))

    # Prints current imaginary time
    if np.mod(i, Nframe) == 0:
        print('it = %1.4f' % t)

    t += dt

# ----------------------------------------------------------------------------------------------------------------------
# Real time evolution
# ----------------------------------------------------------------------------------------------------------------------
t = 0   # Reset time

for i in range(Nt):

    # Forward FFTs
    fft_forward(psi_plus, psi_plus_k)
    fft_forward(psi_0, psi_0_k)
    fft_forward(psi_minus, psi_minus_k)

    # Computing kinetic energy + quadratic Zeeman
    psi_plus_k *= exp(-0.25 * 1j * dt * (Kx ** 2 + Ky ** 2 + 2 * q))
    psi_0_k *= exp(-0.25 * 1j * dt * (Kx ** 2 + Ky ** 2))
    psi_minus_k *= exp(-0.25 * 1j * dt * (Kx ** 2 + Ky ** 2 + 2 * q))

    # Inverse FFTs
    fft_backward(psi_plus_k, psi_plus)
    fft_backward(psi_0_k, psi_0)
    fft_backward(psi_minus_k, psi_minus)

    # Spin vector terms:
    F_perp = sqrt(2.) * (conj(psi_plus) * psi_0 + conj(psi_0) * psi_minus)
    Fz = abs(psi_plus) ** 2 - abs(psi_minus) ** 2
    F = sqrt(abs(Fz) ** 2 + abs(F_perp) ** 2)  # Magnitude of spin vector

    # Total density
    n = abs(psi_minus) ** 2 + abs(psi_0) ** 2 + abs(psi_plus) ** 2

    # Sin and cosine terms for solution
    C = ne.evaluate("cos(c1 * F * dt)")
    S = ne.evaluate("1j * sin(c1 * F * dt) / F")
    S = np.nan_to_num(S)  # Corrects division by 0

    psi_plus_old = psi_plus
    psi_0_old = psi_0
    psi_minus_old = psi_minus

    # Trap, linear Zeeman & interaction flow
    psi_plus = ne.evaluate("C * psi_plus_old - (S * (Fz * psi_plus_old + conj(F_perp) / sqrt(2) * psi_0_old))")
    psi_0 = ne.evaluate("C * psi_0_old - (S / sqrt(2) * (F_perp * psi_plus_old + conj(F_perp) * psi_minus_old))")
    psi_minus = ne.evaluate("C * psi_minus_old - (S * (F_perp / sqrt(2) * psi_0_old - Fz * psi_minus_old))")

    psi_plus *= ne.evaluate("exp(-1j * (dt * (V - p + c0 * n)))")
    psi_0 *= ne.evaluate("exp(-1j * (dt * (V + c0 * n)))")
    psi_minus *= ne.evaluate("exp(-1j * (dt * (V + p + c0 * n)))")

    # Forward FFTs
    fft_forward(psi_plus, psi_plus_k)
    fft_forward(psi_0, psi_0_k)
    fft_forward(psi_minus, psi_minus_k)

    # Computing kinetic energy + quadratic Zeeman
    psi_plus_k *= exp(-0.25 * 1j * dt * (Kx ** 2 + Ky ** 2 + 2 * q))
    psi_0_k *= exp(-0.25 * 1j * dt * (Kx ** 2 + Ky ** 2))
    psi_minus_k *= exp(-0.25 * 1j * dt * (Kx ** 2 + Ky ** 2 + 2 * q))

    # Inverse FFTs
    fft_backward(psi_plus_k, psi_plus)
    fft_backward(psi_0_k, psi_0)
    fft_backward(psi_minus_k, psi_minus)

    # Saves data to hdf5 file
    if np.mod(i + 1, Nframe) == 0:
        # Updates file with new wavefunction values:
        with h5py.File(data_path, 'r+') as data:
            new_psi_plus = data['wavefunction/psi_plus']
            new_psi_plus.resize((Nx, Ny, k + 1))
            new_psi_plus[:, :, k] = psi_plus

            new_psi_0 = data['wavefunction/psi_0']
            new_psi_0.resize((Nx, Ny, k + 1))
            new_psi_0[:, :, k] = psi_0

            new_psi_minus = data['wavefunction/psi_minus']
            new_psi_minus.resize((Nx, Ny, k + 1))
            new_psi_minus[:, :, k] = psi_minus

        k += 1  # Increment array index

    # Print current time
    if np.mod(i, Nframe) == 0:
        print('t = %1.4f' % t)
    t += dt
