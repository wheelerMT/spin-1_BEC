import h5py
import numpy as np
import matplotlib.pyplot as plt
from numpy import conj
from numpy.fft import fft, ifft, ifftshift

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

Q_xx = fft(np.real(conj(psi_plus) * psi_minus) - 0.5 * (n_plus + n_minus) + n / 3)
Q_yy = fft(-np.real(conj(psi_plus) * psi_minus) - 0.5 * (n_plus + n_minus) + n / 3)
Q_zz = fft(-n_0 + n / 3)
Q_xy = fft(np.imag(conj(psi_plus) * psi_minus))
Q_xz = fft(-np.sqrt(2.) / 4 * (psi_0 * (conj(psi_minus - psi_minus)) + conj(psi_0) * (psi_minus - psi_plus)))
Q_yz = fft(-1j * np.sqrt(2.) / 4 * (psi_0 * (conj(psi_minus + psi_minus)) - conj(psi_0) * (psi_minus + psi_plus)))

# Calculate Q_tilde
Qt_xx = (1 / Nx * ifftshift(ifft(Q_xx * conj(Q_xx) + Q_xy * conj(Q_xy) + Q_xz * conj(Q_xz)))).real
Qt_xy = (1 / Nx * ifftshift(ifft(Q_xx * conj(Q_xy) + Q_xy * conj(Q_yy) + Q_xz * conj(Q_yz)))).real
Qt_xz = (1 / Nx * ifftshift(ifft(Q_xx * conj(Q_xz) + Q_xy * conj(Q_yz) + Q_xz * conj(Q_zz)))).real

Qt_yx = (1 / Nx * ifftshift(ifft(Q_xy * conj(Q_xx) + Q_yy * conj(Q_xy) + Q_yz * conj(Q_xz)))).real
Qt_yy = (1 / Nx * ifftshift(ifft(Q_xy * conj(Q_xy) + Q_yy * conj(Q_yy) + Q_yz * conj(Q_yz)))).real
Qt_yz = (1 / Nx * ifftshift(ifft(Q_xy * conj(Q_xz) + Q_yy * conj(Q_yz) + Q_yz * conj(Q_zz)))).real

Qt_zx = (1 / Nx * ifftshift(ifft(Q_xz * conj(Q_xx) + Q_yz * conj(Q_xy) + Q_zz * conj(Q_xz)))).real
Qt_zy = (1 / Nx * ifftshift(ifft(Q_xz * conj(Q_xy) + Q_yz * conj(Q_yy) + Q_zz * conj(Q_yz)))).real
Qt_zz = (1 / Nx * ifftshift(ifft(Q_xz * conj(Q_xz) + Q_yz * conj(Q_yz) + Q_zz * conj(Q_zz)))).real

eigenvalues_1 = []
eigenvalues_2 = []
eigenvalues_3 = []
eigenvector_1 = []
eigenvector_2 = []
eigenvector_3 = []
for index in range(Nx):
    Qt = np.matrix([[Qt_xx[index], Qt_xy[index], Qt_xz[index]],
                    [Qt_yx[index], Qt_yy[index], Qt_yz[index]],
                    [Qt_zx[index], Qt_zy[index], Qt_zz[index]]])
    w, v = np.linalg.eig(Qt)

    eigenvalues_1.append(w[0])
    eigenvector_1.append(v[0])
    eigenvalues_2.append(w[1])
    eigenvector_2.append(v[1])
    eigenvalues_3.append(w[2])
    eigenvector_3.append(v[2])

# plt.plot(x[-Nx // 2 - 100:Nx // 2 + 100], eigenvalues[-Nx // 2 - 100:Nx // 2 + 100], 'ko')
plt.plot(x, eigenvalues_1, 'ko')
plt.plot(x, eigenvalues_2, 'ro')
plt.plot(x, eigenvalues_3, 'bo')

# print(eigenvector_1[np.where(np.array(eigenvalues_1) > 0.1)[0][0]])
# plt.savefig(f'../../../../plots/spin-1/{filename_prefix}_eigenvalue.png', bbox_inches='tight')
plt.ylim(0, 0.3)
plt.show()
