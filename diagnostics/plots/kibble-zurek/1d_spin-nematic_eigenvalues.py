import h5py
import numpy as np
import matplotlib.pyplot as plt
from numpy import conj

# Load in data
filename_prefix = '1d_polar-BA-FM_5000'
data_file = h5py.File('../../../data/1d_kibble-zurek/{}.hdf5'.format(filename_prefix), 'r')

# Loading grid array data:
x = data_file['grid/x'][...]
Nx = len(x)
dx = x[1] - x[0]
dkx = np.pi / (Nx / 2 * dx)
kx = np.arange(-Nx // 2, Nx // 2) * dkx
box_radius = int(np.ceil(np.sqrt(Nx ** 2) / 2) + 1)
center_x = Nx // 2

time = data_file['time/t'][:, 0]

# Generate figure
fig, ax = plt.subplots()
# ax.set_ylim(0, 1.1)
ax.set_ylabel(r'$\lambda_Q (r)$')
ax.set_xlabel(r'$x / \xi_s$')

# Loading wavefunction data
psi_plus = data_file['wavefunction/psi_plus'][:, -1]
psi_0 = data_file['wavefunction/psi_0'][:, -1]
psi_minus = data_file['wavefunction/psi_minus'][:, -1]

# Calculate densities
n_plus = abs(psi_plus) ** 2
n_0 = abs(psi_0) ** 2
n_minus = abs(psi_minus) ** 2
n = abs(psi_plus) ** 2 + abs(psi_0) ** 2 + abs(psi_minus) ** 2

Q_xx = np.real(conj(psi_plus) * psi_minus) \
       - 0.5 * (n_plus + n_minus) + n / 3
Q_yy = -np.real(conj(psi_plus) * psi_minus) - 0.5 * (n_plus + n_minus) + n / 3
Q_zz = -n_0 + n / 3
Q_xy = np.imag(conj(psi_plus) * psi_minus)
Q_xz = -np.sqrt(2.) / 4 * (psi_0 * (conj(psi_minus - psi_minus))
                           + conj(psi_0) * (psi_minus - psi_plus))
Q_yz = -1j * np.sqrt(2.) / 4 * (psi_0 * (conj(psi_minus + psi_minus))
                                - conj(psi_0) * (psi_minus + psi_plus))

Q_xx0 = np.roll(Q_xx, Nx // 3)
Q_yy0 = np.roll(Q_yy, Nx // 3)
Q_zz0 = np.roll(Q_zz, Nx // 3)
Q_xy0 = np.roll(Q_xy, Nx // 3)
Q_xz0 = np.roll(Q_xz, Nx // 3)
Q_yz0 = np.roll(Q_yz, Nx // 3)

# Calculate Q_tilde
Qt_xx = Q_xx0 * Q_xx + Q_xy0 * Q_xy + Q_xz0 * Q_xz
Qt_xy = Q_xx0 * Q_xy + Q_xy0 * Q_yy + Q_xz0 * Q_yz
Qt_xz = Q_xx0 * Q_xz + Q_xy0 * Q_yz + Q_xz0 * Q_zz

Qt_yx = Q_xy0 * Q_xx + Q_yy0 * Q_xy + Q_yz0 * Q_xz
Qt_yy = Q_xy0 * Q_xy + Q_yy0 * Q_yy + Q_yz0 * Q_yz
Qt_yz = Q_xy0 * Q_xz + Q_yy0 * Q_yz + Q_yz0 * Q_zz

Qt_zx = Q_xz0 * Q_xx + Q_yz0 * Q_xy + Q_zz0 * Q_xz
Qt_zy = Q_xz0 * Q_xy + Q_yz0 * Q_yy + Q_zz0 * Q_yz
Qt_zz = Q_xz0 * Q_xz + Q_yz0 * Q_yz + Q_zz0 * Q_zz

eigenvalues = []
for index in range(Nx):
    Qt = np.matrix([[Qt_xx[index], Qt_xy[index], Qt_xz[index]],
                    [Qt_yx[index], Qt_yy[index], Qt_yz[index]],
                    [Qt_zx[index], Qt_zy[index], Qt_zz[index]]])
    w, v = np.linalg.eig(Qt)

    eigenvalues.append(w[0])

# plt.plot(x[-Nx // 2 - 100:Nx // 2 + 100], eigenvalues[-Nx // 2 - 100:Nx // 2 + 100], 'ko')
plt.plot(x, eigenvalues, 'ko')
plt.savefig(f'../../../../plots/spin-1/{filename_prefix}_eigenvalue.png', bbox_inches='tight')
plt.show()
