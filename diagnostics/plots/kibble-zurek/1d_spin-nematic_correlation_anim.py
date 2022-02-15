import h5py
import numpy as np
from matplotlib import pyplot as plt, animation
from numpy import conj
from numpy.fft import fft, ifft, ifftshift


def animate(i):
    print(f'On frame {i}')

    # Loading wavefunction data
    psi_plus = data_file['wavefunction/psi_plus'][:, i]
    psi_0 = data_file['wavefunction/psi_0'][:, i]
    psi_minus = data_file['wavefunction/psi_minus'][:, i]

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

    G_phi = 1 / Nx * ifftshift(ifft(Q_xx * conj(Q_xx) + Q_yy * conj(Q_yy) + Q_zz * conj(Q_zz) +
                                    2 * (Q_xy * conj(Q_xy) + Q_xz * conj(Q_xz) + Q_yz * conj(Q_yz))))

    # Re-calculate correlation function for density contributions
    Q_xx = fft(-0.5 * (n_plus + n_minus) + n / 3)
    Q_yy = fft(-0.5 * (n_plus + n_minus) + n / 3)
    Q_zz = fft(-n_0 + n / 3)
    G_phi_dens = 1 / Nx * ifftshift(ifft(Q_xx * conj(Q_xx) + Q_yy * conj(Q_yy) + Q_zz * conj(Q_zz)))

    G_phi_line.set_ydata(G_phi.flatten())
    G_phi_dens_line.set_ydata(G_phi_dens.flatten())
    G_phi_subtract_line.set_ydata((G_phi - G_phi_dens).flatten())
    plt.title(f't={int(time[i])}')


# Load in data
filename_prefix = '1d_swislocki_5000'
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
ax.set_ylim(0, 1.1)
ax.set_ylabel(r'$G_\phi(r)$')
ax.set_xlabel(r'$r / \xi_s$')

# Loading wavefunction data
psi_plus = data_file['wavefunction/psi_plus'][:, 0]
psi_0 = data_file['wavefunction/psi_0'][:, 0]
psi_minus = data_file['wavefunction/psi_minus'][:, 0]

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

G_phi = 1 / Nx * ifftshift(ifft(Q_xx * conj(Q_xx) + Q_yy * conj(Q_yy) + Q_zz * conj(Q_zz) +
                                2 * (Q_xy * conj(Q_xy) + Q_xz * conj(Q_xz) + Q_yz * conj(Q_yz))))

# Re-calculate correlation function for density contributions
Q_xx = fft(-0.5 * (n_plus + n_minus) + n / 3)
Q_yy = fft(-0.5 * (n_plus + n_minus) + n / 3)
Q_zz = fft(-n_0 + n / 3)
G_phi_dens = 1 / Nx * ifftshift(ifft(Q_xx * conj(Q_xx) + Q_yy * conj(Q_yy) + Q_zz * conj(Q_zz)))

# Do initial plot
G_phi_line, G_phi_dens_line, G_phi_subtract_line = plt.plot(x, G_phi, 'k', x, G_phi_dens, 'r', x,
                                                            G_phi - G_phi_dens, 'b')
G_phi_line.set_label('Full correlation function')
G_phi_dens_line.set_label('Only diagonal density')
G_phi_subtract_line.set_label('Subtracted')
ax.legend()

anim = animation.FuncAnimation(fig, animate, frames=1000)
anim.save('../../../../plots/spin-1/{}_g_phi.mp4'.format(filename_prefix), writer=animation.FFMpegWriter(fps=30))
