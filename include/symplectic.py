import cupy as cp

"""Module file that contains necessary functions for solving the spin-1 GPE using a symplectic method. These functions
use CuPy to wrap around CUDA for a fast and efficient evolution."""


def rotation(wfn, Nx, Ny, alpha, beta, gamma):
    """Function that performs a spin rotation with angles alpha, beta and gamma to a three-component wavefunction."""
    wfn_new = cp.empty((3, Nx, Ny), dtype='complex64')
    U = cp.empty((3, 3), dtype='complex64')

    # Spin-1 rotation matrix
    U[0, 0] = cp.exp(-1j * (alpha + gamma)) * (cp.cos(beta / 2)) ** 2
    U[0, 1] = -cp.exp(-1j * alpha) / cp.sqrt(2.) * cp.sin(beta)
    U[0, 2] = cp.exp(-1j * (alpha - gamma)) * (cp.sin(beta / 2)) ** 2
    U[1, 0] = cp.exp(-1j * gamma) / cp.sqrt(2.) * cp.sin(beta)
    U[1, 1] = cp.cos(beta)
    U[1, 2] = -cp.exp(1j * gamma) / cp.sqrt(2) * cp.sin(beta)
    U[2, 0] = cp.exp(1j * (alpha - gamma)) * (cp.sin(beta / 2)) ** 2
    U[2, 1] = cp.exp(1j * alpha) / cp.sqrt(2.) * cp.sin(beta)
    U[2, 2] = cp.exp(1j * (alpha + gamma)) * (cp.cos(beta / 2)) ** 2

    wfn_new[0, :, :] = U[0, 0] * wfn[0, :, :] + U[0, 1] * wfn[1, :, :] + U[0, 2] * wfn[2, :, :]
    wfn_new[1, :, :] = U[1, 0] * wfn[0, :, :] + U[1, 1] * wfn[1, :, :] + U[1, 2] * wfn[2, :, :]
    wfn_new[2, :, :] = U[2, 0] * wfn[0, :, :] + U[2, 1] * wfn[1, :, :] + U[2, 2] * wfn[2, :, :]
    return wfn_new[0, :, :], wfn_new[1, :, :], wfn_new[2, :, :]


def fourier_space(wfn_plus, wfn_0, wfn_minus, dt, Kx, Ky, q):
    """Solves the kinetic energy term in Fourier space."""
    wfn_plus *= cp.exp(-0.25 * 1j * dt * (Kx ** 2 + Ky ** 2 + 2 * q))
    wfn_0 *= cp.exp(-0.25 * 1j * dt * (Kx ** 2 + Ky ** 2))
    wfn_minus *= cp.exp(-0.25 * 1j * dt * (Kx ** 2 + Ky ** 2 + 2 * q))


def fourier_space_1d(wfn_plus, wfn_0, wfn_minus, dt, Kx, q):
    """Solves the kinetic energy term in Fourier space."""
    wfn_plus *= cp.exp(-0.25 * 1j * dt * (Kx ** 2 + 2 * q))
    wfn_0 *= cp.exp(-0.25 * 1j * dt * (Kx ** 2))
    wfn_minus *= cp.exp(-0.25 * 1j * dt * (Kx ** 2 + 2 * q))


def fourier_space_KZ_1d(wfn_plus, wfn_0, wfn_minus, dt, Kx, t, tau_q, c2, n_0):
    """Solves the kinetic energy and time-dependent quadratic Zeeman
    term in Fourier space."""
    wfn_plus *= cp.exp(-0.25 * 1j * dt * (Kx ** 2 + 2 * abs(c2) * n_0 * (2 - t * (1 + dt) / (2 * tau_q))))
    wfn_0 *= cp.exp(-0.25 * 1j * dt * (Kx ** 2))
    wfn_minus *= cp.exp(-0.25 * 1j * dt * (Kx ** 2 + 2 * abs(c2) * n_0 * (2 - t * (1 + dt) / (2 * tau_q))))


def calc_spin_dens(wfn_plus, wfn_0, wfn_minus, dt, c2):
    """Calculates various quantities such as spin vectors, sin and cosine terms and the atomic density."""
    spin_perp = cp.sqrt(2.) * (cp.conj(wfn_plus) * wfn_0 + cp.conj(wfn_0) * wfn_minus)
    spin_z = cp.abs(wfn_plus) ** 2 - cp.abs(wfn_minus) ** 2
    F = cp.sqrt(cp.abs(spin_z) ** 2 + cp.abs(spin_perp) ** 2)  # Magnitude of spin vector

    cos_term = cp.cos(c2 * F * dt)
    sin_term = 1j * cp.sin(c2 * F * dt) / F
    sin_term = cp.nan_to_num(sin_term)  # Corrects division by 0

    density = cp.abs(wfn_minus) ** 2 + cp.abs(wfn_0) ** 2 + cp.abs(wfn_plus) ** 2

    return spin_perp, spin_z, cos_term, sin_term, density


def transverse_mag(wfn_plus, wfn_0, wfn_minus, dx):
    dens = abs(wfn_plus) ** 2 + abs(wfn_0) ** 2 + abs(wfn_minus) ** 2
    spin_perp = cp.sqrt(2.) * (cp.conj(wfn_plus) * wfn_0 + cp.conj(wfn_0) * wfn_minus)

    return dx * cp.sum(abs(spin_perp) ** 2 / dens)


def interaction_flow(wfn_plus, wfn_0, wfn_minus, C, S, Fz, F_perp, dt, V, p, c0, n):
    """Solves the interaction part of the flow."""
    new_wfn_plus = (C * wfn_plus - S * (Fz * wfn_plus + cp.conj(F_perp) / cp.sqrt(2) * wfn_0)) * cp.exp(
        -1j * dt * (V - p + c0 * n))
    new_wfn_0 = (C * wfn_0 - S / cp.sqrt(2) * (F_perp * wfn_plus + cp.conj(F_perp) * wfn_minus)) * cp.exp(
        -1j * dt * (V + c0 * n))
    new_wfn_minus = (C * wfn_minus - S * (F_perp / cp.sqrt(2) * wfn_0 - Fz * wfn_minus)) * cp.exp(
        -1j * dt * (V + p + c0 * n))

    return new_wfn_plus, new_wfn_0, new_wfn_minus


def renorm_mag(wfn_plus, wfn_0, wfn_minus, target_mag):
    N_plus = cp.sum(abs(wfn_plus) ** 2)
    N_0 = cp.sum(abs(wfn_0) ** 2)
    N_minus = cp.sum(abs(wfn_minus) ** 2)

    N = N_plus + N_0 + N_minus

    # Correction factors
    r_plus = cp.sqrt((1 + (target_mag * N + N_minus - N_plus) / (2 * N_plus)) / N)
    r_0 = cp.sqrt(1 / N)
    r_minus = cp.sqrt((1 - (target_mag * N + N_minus - N_plus) / (2 * N_minus)) / N)

    wfn_plus *= r_plus
    wfn_0 *= r_0
    wfn_minus *= r_minus
