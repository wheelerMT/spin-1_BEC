import h5py
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.animation as animation

"""File that calculates the divergence of the spin current for an polar spin-1 condensate in the EPP phase."""


def spectral_derivative(array, wvn):
    return (ifft2(ifftshift(1j * wvn * fftshift(fft2(array))))).real


# ---------------------------------------------------------------------------------------------------------------------
# Loading data
# ---------------------------------------------------------------------------------------------------------------------
filename = 'dipole/HQV_gamma075_AP'  # input('Enter filename: ')
data_path = '../../data/{}.hdf5'.format(filename)
data_file = h5py.File(data_path, 'r')

# Loading grid array data:
x, y = data_file['grid/x'], data_file['grid/y']
X, Y = np.meshgrid(x[:], y[:])
Nx, Ny = x[:].size, y[:].size
dx, dy = x[1] - x[0], y[1] - y[0]
dkx, dky = 2 * np.pi / (Nx * dx), 2 * np.pi / (Ny * dy)
kxx = np.arange(-Nx // 2, Nx // 2) * dkx
kyy = np.arange(-Nx // 2, Nx // 2) * dky
Kx, Ky = np.meshgrid(kxx, kyy)

# Loading time variables:
Nt, dt, Nframe = np.array(data_file['time/Nt']), np.array(data_file['time/dt']), np.array(data_file['time/Nframe'])

# Three component wavefunction
psi_plus = data_file['wavefunction/psi_plus']
psi_0 = data_file['wavefunction/psi_0']
psi_minus = data_file['wavefunction/psi_minus']

num_of_frames = psi_plus.shape[-1]

# ---------------------------------------------------------------------------------------------------------------------
# Setting up figure
# ---------------------------------------------------------------------------------------------------------------------
# Set up figure:
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(12, 10))
ax[0].set_ylabel(r'$y / \xi_s$')
ax[0].set_title(r'$|\psi_+|^2$')
ax[1].set_title(r'$\nabla \cdot n\vec{v}^\mathrm{spin}_z$')
for axis in ax:
    axis.set_xlabel(r'$x / \xi_s$')

cvals_dens = np.linspace(0, 1, 25, endpoint=True)
cvals_div = np.linspace(-0.5, 0.5, 25, endpoint=True)

# Calculate divergence of spin current for initial plot:
dpsi_plus_x = spectral_derivative(psi_plus[:, :, 0], Kx)
dpsi_plus_y = spectral_derivative(psi_plus[:, :, 0], Ky)
dpsi_minus_x = spectral_derivative(psi_minus[:, :, 0], Kx)
dpsi_minus_y = spectral_derivative(psi_minus[:, :, 0], Ky)

nvz_spin_x = 1 / 2j * (np.conj(psi_plus[:, :, 0]) * dpsi_plus_x - np.conj(dpsi_plus_x) * psi_plus[:, :, 0]
                       - (np.conj(psi_minus[:, :, 0]) * dpsi_minus_x - np.conj(dpsi_minus_x) * psi_minus[:, :, 0]))
nvz_spin_y = 1 / 2j * (np.conj(psi_plus[:, :, 0]) * dpsi_plus_y - np.conj(dpsi_plus_y) * psi_plus[:, :, 0]
                       - (np.conj(psi_minus[:, :, 0]) * dpsi_minus_y - np.conj(dpsi_minus_y) * psi_minus[:, :, 0]))

div_spinz = spectral_derivative(nvz_spin_x, Kx) + spectral_derivative(nvz_spin_y, Ky)

densPlus_plot = ax[0].contourf(X[Nx//4:3*Nx//4, :], Y[Nx//4:3*Nx//4, :], abs(psi_plus[Nx//4:3*Nx//4, :, 0]) ** 2, cvals_dens, cmap='gnuplot')
div_plot = ax[1].contourf(X[Nx//4:3*Nx//4, :], Y[Nx//4:3*Nx//4, :], div_spinz[Nx//4:3*Nx//4, :], cvals_div, cmap='seismic')
cont = [densPlus_plot, div_plot]


# Set up color bars:
dens_cbar = plt.colorbar(densPlus_plot, ax=ax[0], ticks=[0, 1], fraction=0.027, pad=0.03)
phase_cbar = plt.colorbar(div_plot, ax=ax[1], ticks=[-0.5, 0.5], fraction=0.027, pad=0.03)

for axis in ax:
    axis.set_aspect('equal')


# ---------------------------------------------------------------------------------------------------------------------
# Animation function
# ---------------------------------------------------------------------------------------------------------------------


def animate(i):
    """Animation function for plots."""
    global cont
    for contour in cont:
        for c in contour.collections:
            c.remove()

    ax[0].contourf(X[Nx//4:3*Nx//4, :], Y[Nx//4:3*Nx//4, :], abs(psi_plus[Nx//4:3*Nx//4, :, i]) ** 2, cvals_dens, cmap='gnuplot')

    # Calculate divergence of spin current:
    dpsi_plus_x = spectral_derivative(psi_plus[:, :, i], Kx)
    dpsi_plus_y = spectral_derivative(psi_plus[:, :, i], Ky)
    dpsi_minus_x = spectral_derivative(psi_minus[:, :, i], Kx)
    dpsi_minus_y = spectral_derivative(psi_minus[:, :, i], Ky)

    nvz_spin_x = 1 / 2j * (np.conj(psi_plus[:, :, i]) * dpsi_plus_x - np.conj(dpsi_plus_x) * psi_plus[:, :, i]
                           - (np.conj(psi_minus[:, :, i]) * dpsi_minus_x - np.conj(dpsi_minus_x) * psi_minus[:, :, i]))
    nvz_spin_y = 1 / 2j * (np.conj(psi_plus[:, :, i]) * dpsi_plus_y - np.conj(dpsi_plus_y) * psi_plus[:, :, i]
                           - (np.conj(psi_minus[:, :, i]) * dpsi_minus_y - np.conj(dpsi_minus_y) * psi_minus[:, :, i]))

    div_spinz = spectral_derivative(nvz_spin_x, Kx) + spectral_derivative(nvz_spin_y, Ky)

    ax[1].contourf(X[Nx//4:3*Nx//4, :], Y[Nx//4:3*Nx//4, :], div_spinz[Nx//4:3*Nx//4, :], cvals_div, cmap='seismic')

    cont = [ax[0], ax[1]]
    print('On density iteration %i' % (i + 1))
    plt.suptitle(r'$\tau$ = %2f' % (Nframe * dt * i), y=0.7)
    return cont


# Calls the animation function and saves the result
anim = animation.FuncAnimation(fig, animate, frames=num_of_frames, repeat=False)
anim.save('../../../plots/spin-1/spinDiv_{}.mp4'.format(filename[7:]), dpi=200,
          writer=animation.FFMpegWriter(fps=60, codec="libx264", extra_args=['-pix_fmt', 'yuv420p']))
