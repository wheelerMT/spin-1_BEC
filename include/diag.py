import numpy as np
import numexpr as ne
from numpy.fft import fftshift, ifftshift

""" File that contains many useful functions for calculating various quantities of spinor BECs."""


def calculate_density(wfn_plus, wfn_0, wfn_minus):
    return ne.evaluate("abs(wfn_plus).real ** 2 + abs(wfn_0).real ** 2 + abs(wfn_minus).real ** 2")


def calculate_spin(wfn_plus, wfn_0, wfn_minus, dens):
    spin_x = ne.evaluate("1 / sqrt(2) * (conj(wfn_plus) * wfn_0 + conj(wfn_0) * (wfn_plus + wfn_minus) "
                         "+ conj(wfn_minus) * wfn_0)").real
    spin_y = ne.evaluate("1j / sqrt(2) * (-conj(wfn_plus) * wfn_0 + conj(wfn_0) * (wfn_plus - wfn_minus) "
                         "+ conj(wfn_minus) * wfn_0)").real
    spin_z = ne.evaluate("abs(wfn_plus).real ** 2 - abs(wfn_minus).real ** 2")

    expectation = ne.evaluate("sqrt(spin_x ** 2 + spin_y ** 2 + spin_z ** 2) / dens")

    return spin_x, spin_y, spin_z, expectation


def spectral_derivative(array, wvn_x, wvn_y, fft2=np.fft.fft2, ifft2=np.fft.ifft2):
    return ifft2(ifftshift(1j * wvn_x * fftshift(fft2(array)))), ifft2(ifftshift(1j * wvn_y * fftshift(fft2(array))))


def spectral_derivative_1d(array, wvn_x):
    return np.fft.ifft(1j * wvn_x * np.fft.fft(array))


def calculate_mass_current(wfn_plus, wfn_0, wfn_minus, dpsi_p_x, dpsi_p_y, dpsi_0_x, dpsi_0_y, dpsi_m_x, dpsi_m_y):
    mass_cur_x = ne.evaluate("(conj(wfn_minus) * dpsi_m_x - conj(dpsi_m_x) * wfn_minus "
                             "+ conj(wfn_0) * dpsi_0_x - conj(dpsi_0_x) * wfn_0"
                             "+ conj(wfn_plus) * dpsi_p_x - conj(dpsi_p_x) * wfn_plus) / 2j").real

    mass_cur_y = ne.evaluate("(conj(wfn_minus) * dpsi_m_y - conj(dpsi_m_y) * wfn_minus "
                             "+ conj(wfn_0) * dpsi_0_y - conj(dpsi_0_y) * wfn_0"
                             "+ conj(wfn_plus) * dpsi_p_y - conj(dpsi_p_y) * wfn_plus) / 2j").real

    return mass_cur_x, mass_cur_y


def calculate_pseudo_vorticity(mass_current_x, mass_current_y, dxx, dyy):
    return np.gradient(mass_current_y, dxx, axis=0) - np.gradient(mass_current_x, dyy, axis=1)


def calculate_scalar_energy(wfn, Kx, Ky, m, V, g):
    dwfn_x, dwfn_y = spectral_derivative(wfn, Kx, Ky)
    kinetic_energy = np.sum(1 / (2 * m) * (abs(dwfn_x) ** 2 + abs(dwfn_y) ** 2))

    potential_energy = np.sum(V * abs(wfn) ** 2)

    interaction_energy = np.sum(g / 2 * abs(wfn) ** 4)

    return kinetic_energy + potential_energy + interaction_energy


def calculate_energy_1d(wfn_plus, wfn_0, wfn_minus, dx, wvn_x, m, p, q, V, c0, c2):
    psi = [wfn_plus, wfn_0, wfn_minus]
    n = calculate_density(wfn_plus, wfn_0, wfn_minus)
    fx, fy, fz, _ = calculate_spin(wfn_plus, wfn_0, wfn_minus, n)
    F2 = abs(fx) ** 2 + abs(fy) ** 2 + fz ** 2

    # Calculate kinetic energy
    dwfn_plus = spectral_derivative_1d(wfn_plus, wvn_x)
    dwfn_0 = spectral_derivative_1d(wfn_0, wvn_x)
    dwfn_minus = spectral_derivative_1d(wfn_minus, wvn_x)
    kinetic_energy = dx * np.sum(1 / (2 * m) * (abs(dwfn_plus) ** 2 + abs(dwfn_0) ** 2 + abs(dwfn_minus) ** 2))

    # Calculate potential energy
    pot_energy = 0
    for i, wfn in enumerate(psi):
        mF = 1 - i  # Spin component
        pot_energy += (V - p * mF + q * mF ** 2) * abs(wfn) ** 2
    pot_energy = dx * np.sum(pot_energy)

    # Calculate interaction energy
    interaction_energy = dx * np.sum(c0 / 2 * n ** 2 + c2 / 2 * F2)

    return kinetic_energy, pot_energy, interaction_energy
