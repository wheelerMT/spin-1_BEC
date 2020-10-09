import h5py
import numpy as np
import numexpr as ne

"""File that checks conservation of atom number and magnetisation for a given dataset."""


def magnetisation(wfn_plus, wfn_0, wfn_minus, dxx=1, dyy=1):
    """Calculates transverse and longitudinal magnetisation for a spinor condensate."""

    # Calculates spin vectors:
    spin_x = ne.evaluate("1 / sqrt(2) * (conj(wfn_plus) * wfn_0 + conj(wfn_0) * (wfn_plus + wfn_minus) "
                         "+ conj(wfn_minus) * wfn_0)").real
    spin_y = ne.evaluate("1j / sqrt(2) * (-conj(wfn_plus) * wfn_0 + conj(wfn_0) * (wfn_plus - wfn_minus) "
                         "+ conj(wfn_minus) * wfn_0)").real
    spin_z = ne.evaluate("abs(wfn_plus).real ** 2 - abs(wfn_minus).real ** 2")

    mag_x = dxx * dyy * np.sum(spin_x)
    mag_y = dxx * dyy * np.sum(spin_y)
    mag_z = dxx * dyy * np.sum(spin_z)

    print(mag_z)

    return mag_x, mag_y, mag_z


def atom_number(wfn_plus, wfn_0, wfn_minus, dxx=1, dyy=1):
    """Calculates atom number of each component of a spinor wavefunction."""

    atom_N_plus = dxx * dyy * np.sum(abs(wfn_plus) ** 2)
    atom_N_0 = dxx * dyy * np.sum(abs(wfn_0) ** 2)
    atom_N_minus = dxx * dyy * np.sum(abs(wfn_minus) ** 2)
    print('Atom number = {:.2e}'.format(atom_N_plus + atom_N_0 + atom_N_minus))

    return atom_N_plus, atom_N_0, atom_N_minus


# Load in data:
filename = input('Enter name of data file: ')
data_file = h5py.File('../data/{}.hdf5'.format(filename), 'r')

x, y = data_file['grid/x'], data_file['grid/y']
X, Y = np.meshgrid(x, y)
Nx, Ny = x.size, y.size
dx, dy = x[1] - x[0], y[1] - y[0]

# Wavefunction data
psi_plus = data_file['wavefunction/psi_plus']
psi_0 = data_file['wavefunction/psi_0']
psi_minus = data_file['wavefunction/psi_minus']

for i in range(psi_plus.shape[-1]):
    atom_number(psi_plus[:, :, i], psi_0[:, :, i], psi_minus[:, :, i])
