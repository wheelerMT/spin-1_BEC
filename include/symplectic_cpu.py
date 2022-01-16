import numpy as np
from numba import njit


"""Module file that contains necessary functions for solving the spin-1 GPE using a symplectic method."""


def rotation(wfn, Nx, Ny, alpha, beta, gamma):
    """Function that performs a spin rotation with angles alpha, beta and gamma to a three-component wavefunction."""
    wfn_new = np.empty((3, Nx, Ny), dtype='complex64')
    U = np.empty((3, 3), dtype='complex64')

    # Spin-1 rotation matrix
    U[0, 0] = np.exp(-1j * (alpha + gamma)) * (np.cos(beta / 2)) ** 2
    U[0, 1] = -np.exp(-1j * alpha) / np.sqrt(2.) * np.sin(beta)
    U[0, 2] = np.exp(-1j * (alpha - gamma)) * (np.sin(beta / 2)) ** 2
    U[1, 0] = np.exp(-1j * gamma) / np.sqrt(2.) * np.sin(beta)
    U[1, 1] = np.cos(beta)
    U[1, 2] = -np.exp(1j * gamma) / np.sqrt(2) * np.sin(beta)
    U[2, 0] = np.exp(1j * (alpha - gamma)) * (np.sin(beta / 2)) ** 2
    U[2, 1] = np.exp(1j * alpha) / np.sqrt(2.) * np.sin(beta)
    U[2, 2] = np.exp(1j * (alpha + gamma)) * (np.cos(beta / 2)) ** 2

    wfn_new[0, :, :] = U[0, 0] * wfn[0, :, :] + U[0, 1] * wfn[1, :, :] + U[0, 2] * wfn[2, :, :]
    wfn_new[1, :, :] = U[1, 0] * wfn[0, :, :] + U[1, 1] * wfn[1, :, :] + U[1, 2] * wfn[2, :, :]
    wfn_new[2, :, :] = U[2, 0] * wfn[0, :, :] + U[2, 1] * wfn[1, :, :] + U[2, 2] * wfn[2, :, :]
    return wfn_new[0, :, :], wfn_new[1, :, :], wfn_new[2, :, :]


@njit
def fourier_space(wfn_plus, wfn_0, wfn_minus, dt, Kx, Ky, q):
    """Solves the kinetic energy term in Fourier space."""
    wfn_plus *= np.exp(-0.25 * 1j * dt * (Kx ** 2 + Ky ** 2 + 2 * q))
    wfn_0 *= np.exp(-0.25 * 1j * dt * (Kx ** 2 + Ky ** 2))
    wfn_minus *= np.exp(-0.25 * 1j * dt * (Kx ** 2 + Ky ** 2 + 2 * q))


@njit
def fourier_space_1d(wfn_plus, wfn_0, wfn_minus, dt, Kx, q):
    """Solves the kinetic energy term in Fourier space."""
    wfn_plus *= np.exp(-0.25 * 1j * dt * (Kx ** 2 + 2 * q))
    wfn_0 *= np.exp(-0.25 * 1j * dt * (Kx ** 2))
    wfn_minus *= np.exp(-0.25 * 1j * dt * (Kx ** 2 + 2 * q))


@njit
def fourier_space_KZ_1d(wfn_plus, wfn_0, wfn_minus, dt, Kx, Q, c2, n_0, tau_q, sign):
    """Solves the kinetic energy and time-dependent quadratic Zeeman
    term in Fourier space."""

    if sign == -1:
        wfn_plus *= np.exp(-0.25 * 1j * dt * (Kx ** 2 + 2 * abs(c2) * n_0 * (Q - dt / (2 * tau_q))))
        wfn_0 *= np.exp(-0.25 * 1j * dt * (Kx ** 2))
        wfn_minus *= np.exp(-0.25 * 1j * dt * (Kx ** 2 + 2 * abs(c2) * n_0 * (Q - dt / (2 * tau_q))))
    elif sign == 1:
        wfn_plus *= np.exp(-0.25 * 1j * dt * (Kx ** 2 + 2 * abs(c2) * n_0 * (Q + dt / (2 * tau_q))))
        wfn_0 *= np.exp(-0.25 * 1j * dt * (Kx ** 2))
        wfn_minus *= np.exp(-0.25 * 1j * dt * (Kx ** 2 + 2 * abs(c2) * n_0 * (Q + dt / (2 * tau_q))))
    else:
        raise ValueError("parameter sign is not 1 or -1.")


@njit
def fourier_space_KZ_2d(wfn_plus, wfn_0, wfn_minus, dt, Kx, Ky, Q, c2, n_0, tau_q, sign):
    """Solves the kinetic energy and time-dependent quadratic Zeeman
    term in Fourier space."""

    # If decreasing Q
    if sign == -1:
        wfn_plus *= np.exp(-0.25 * 1j * dt * (Kx ** 2 + Ky ** 2 + 2 * abs(c2) * n_0 * (Q - dt / (2 * tau_q))))
        wfn_0 *= np.exp(-0.25 * 1j * dt * (Kx ** 2 + Ky ** 2))
        wfn_minus *= np.exp(-0.25 * 1j * dt * (Kx ** 2 + Ky ** 2 + 2 * abs(c2) * n_0 * (Q - dt / (2 * tau_q))))

    # Else, if increasing Q
    elif sign == 1:
        wfn_plus *= np.exp(-0.25 * 1j * dt * (Kx ** 2 + Ky ** 2 + 2 * abs(c2) * n_0 * (Q + dt / (2 * tau_q))))
        wfn_0 *= np.exp(-0.25 * 1j * dt * (Kx ** 2 + Ky ** 2))
        wfn_minus *= np.exp(-0.25 * 1j * dt * (Kx ** 2 + Ky ** 2 + 2 * abs(c2) * n_0 * (Q + dt / (2 * tau_q))))

    else:
        raise ValueError("parameter sign is not 1 or -1.")


@njit
def calc_spin_dens(wfn_plus, wfn_0, wfn_minus, dt, c2):
    """Calculates various quantities such as spin vectors, sin and cosine terms and the atomic density."""
    spin_perp = np.sqrt(2.) * (np.conj(wfn_plus) * wfn_0 + np.conj(wfn_0) * wfn_minus)
    spin_z = np.abs(wfn_plus) ** 2 - np.abs(wfn_minus) ** 2
    F = np.sqrt(np.abs(spin_z) ** 2 + np.abs(spin_perp) ** 2)  # Magnitude of spin vector

    cos_term = np.cos(c2 * F * dt)
    sin_term = 1j * np.sin(c2 * F * dt) / F
    # sin_term[F < 1e-8] = 0  # Corrects division by 0

    density = np.abs(wfn_minus) ** 2 + np.abs(wfn_0) ** 2 + np.abs(wfn_plus) ** 2

    return spin_perp, spin_z, cos_term, sin_term, density


@njit
def transverse_mag(wfn_plus, wfn_0, wfn_minus, dx):
    dens = abs(wfn_plus) ** 2 + abs(wfn_0) ** 2 + abs(wfn_minus) ** 2
    spin_perp = np.sqrt(2.) * (np.conj(wfn_plus) * wfn_0 + np.conj(wfn_0) * wfn_minus)

    return dx * np.sum(abs(spin_perp) ** 2 / dens)


@njit
def interaction_flow(wfn_plus, wfn_0, wfn_minus, C, S, Fz, F_perp, dt, V, p, c0, n):
    """Solves the interaction part of the flow."""
    new_wfn_plus = (C * wfn_plus - S * (Fz * wfn_plus + np.conj(F_perp) / np.sqrt(2) * wfn_0)) * np.exp(
        -1j * dt * (V - p + c0 * n))
    new_wfn_0 = (C * wfn_0 - S / np.sqrt(2) * (F_perp * wfn_plus + np.conj(F_perp) * wfn_minus)) * np.exp(
        -1j * dt * (V + c0 * n))
    new_wfn_minus = (C * wfn_minus - S * (F_perp / np.sqrt(2) * wfn_0 - Fz * wfn_minus)) * np.exp(
        -1j * dt * (V + p + c0 * n))

    return new_wfn_plus, new_wfn_0, new_wfn_minus


def renorm_mag(wfn_plus, wfn_0, wfn_minus, target_mag):
    N_plus = np.sum(abs(wfn_plus) ** 2)
    N_0 = np.sum(abs(wfn_0) ** 2)
    N_minus = np.sum(abs(wfn_minus) ** 2)

    N = N_plus + N_0 + N_minus

    # Correction factors
    r_plus = np.sqrt((1 + (target_mag * N + N_minus - N_plus) / (2 * N_plus)) / N)
    r_0 = np.sqrt(1 / N)
    r_minus = np.sqrt((1 - (target_mag * N + N_minus - N_plus) / (2 * N_minus)) / N)

    wfn_plus *= r_plus
    wfn_0 *= r_0
    wfn_minus *= r_minus