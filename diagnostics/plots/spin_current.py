import h5py
import numpy as np
from tabulate import tabulate
import include.diag as diag
from numpy import conj
import include.vortex_detection as vd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
matplotlib.use('TkAgg')


def animate_curl(i):
    global cont
    for contour in cont:
        for c in contour.collections:
            c.remove()

    # Load in wavefunction:
    psi_plus = data_file['wavefunction/psi_plus'][:, :, i]
    psi_0 = data_file['wavefunction/psi_0'][:, :, i]
    psi_minus = data_file['wavefunction/psi_minus'][:, :, i]
    theta = (np.angle(psi_plus) + np.angle(psi_minus)) / 2

    # Detect vortices:
    # psqv, nsqv, phqv_plus, nhqv_plus, phqv_minus, nhqv_minus, num_of_vortices = \
        # vd.calculate_vortices(psi_plus, psi_minus, x, y)

    # ---------------------------------------------------------------------------------------------------------------------
    # Calculate curl and divergence of spin current:
    # ---------------------------------------------------------------------------------------------------------------------
    # Get derivatives of wavefunction:
    dpsiP_dx, dpsiP_dy = diag.spectral_derivative(psi_plus, Kx, Ky, np.fft.fft2, np.fft.ifft2)
    dpsi0_dx, dpsi0_dy = diag.spectral_derivative(psi_0, Kx, Ky, np.fft.fft2, np.fft.ifft2)
    dpsiM_dx, dpsiM_dy = diag.spectral_derivative(psi_minus, Kx, Ky, np.fft.fft2, np.fft.ifft2)

    # Calculate mass current:
    mass_cur_x, mass_cur_y = diag.calculate_mass_current(psi_plus, psi_0, psi_minus, dpsiP_dx, dpsiP_dy, dpsi0_dx, dpsi0_dy,
                                                         dpsiM_dx, dpsiM_dy)
    mass_cur_x_dx, mass_cur_x_dy = diag.spectral_derivative(mass_cur_x, Kx, Ky, np.fft.fft2, np.fft.ifft2)
    mass_cur_y_dx, mass_cur_y_dy = diag.spectral_derivative(mass_cur_y, Kx, Ky, np.fft.fft2, np.fft.ifft2)

    curl_of_mass_cur = (mass_cur_y_dx - mass_cur_x_dy).real
    # div_mass_cur = (mass_cur_x_dx + mass_cur_y_dy).real

    # Calculate spin current:
    nvz_x = (1 / 2j * (conj(psi_plus) * dpsiP_dx - conj(dpsiP_dx) * psi_plus
                       - (conj(psi_minus) * dpsiM_dx - conj(dpsiM_dx) * psi_minus))).real
    nvz_y = (1 / 2j * (conj(psi_plus) * dpsiP_dy - conj(dpsiP_dy) * psi_plus
                       - (conj(psi_minus) * dpsiM_dy - conj(dpsiM_dy) * psi_minus))).real

    # Calculate the curl:
    nvz_x_dx, nvz_x_dy = diag.spectral_derivative(nvz_x, Kx, Ky, np.fft.fft2, np.fft.ifft2)
    nvz_y_dx, nvz_y_dy = diag.spectral_derivative(nvz_y, Kx, Ky, np.fft.fft2, np.fft.ifft2)

    curl_spin_z = (nvz_y_dx - nvz_x_dy).real
    # div_spin_z = (nvz_x_dx + nvz_y_dy).real

    # Plot curl of spin current
    ax[0].contourf(X, Y, curl_spin_z, curl_spin_cvals, cmap='BrBG')

    r"""
    for axis in ax:
        if len(psqv) != 0:
            axis.plot(*zip(*psqv), 'bo', markersize=5, label=r'$\sigma_{SQV} = 1$')  # Plots positive SQVs
        if len(nsqv) != 0:
            axis.plot(*zip(*nsqv), 'ro', markersize=5, label=r'$\sigma_{SQV} = -1$')  # Plots negative SQVs
        if len(phqv_plus) != 0:
            axis.plot(*zip(*phqv_plus), 'bX', markersize=5, label=r'$\sigma_1 = 1$')  # Positive HQV in psi_plus
        if len(nhqv_plus) != 0:
            axis.plot(*zip(*nhqv_plus), 'rX', markersize=5, label=r'$\sigma_1 = -1$')  # Negative HQV in psi_plus
        if len(phqv_minus) != 0:
            axis.plot(*zip(*phqv_minus), 'b^', markersize=5, label=r'$\sigma_{-1} = 1$')  # Positive HQV in psi_minus
        if len(nhqv_minus) != 0:
            axis.plot(*zip(*nhqv_minus), 'r^', markersize=5, label=r'$\sigma_{-1} = -1$')  # Negative HQV in psi_minus
        axis.legend()
    """

    # Plot density:
    ax[1].contourf(X, Y, abs(psi_plus) ** 2 + abs(psi_minus) ** 2, levels=25, cmap='gnuplot')

    # Plot curl of mass current:
    ax[2].contourf(X, Y, curl_of_mass_cur, curl_mass_cvals, cmap='BrBG')
    plt.suptitle(r'$\tau = {}$'.format(i))

    print('On curl frame {}'.format(i))
    cont = [ax[0], ax[1], ax[2]]


def animate_div(i):
    global cont
    for contour in cont:
        for c in contour.collections:
            c.remove()

    # Load in wavefunction:
    psi_plus = data_file['wavefunction/psi_plus'][:, :, i]
    psi_0 = data_file['wavefunction/psi_0'][:, :, i]
    psi_minus = data_file['wavefunction/psi_minus'][:, :, i]
    # theta = (np.angle(psi_plus) + np.angle(psi_minus)) / 2

    # Detect vortices:
    # psqv, nsqv, phqv_plus, nhqv_plus, phqv_minus, nhqv_minus, num_of_vortices = \
        # vd.calculate_vortices(psi_plus, psi_minus, x, y)

    # ---------------------------------------------------------------------------------------------------------------------
    # Calculate curl and divergence of spin current:
    # ---------------------------------------------------------------------------------------------------------------------
    # Get derivatives of wavefunction:
    dpsiP_dx, dpsiP_dy = diag.spectral_derivative(psi_plus, Kx, Ky, np.fft.fft2, np.fft.ifft2)
    dpsi0_dx, dpsi0_dy = diag.spectral_derivative(psi_0, Kx, Ky, np.fft.fft2, np.fft.ifft2)
    dpsiM_dx, dpsiM_dy = diag.spectral_derivative(psi_minus, Kx, Ky, np.fft.fft2, np.fft.ifft2)

    # Calculate mass current:
    mass_cur_x, mass_cur_y = diag.calculate_mass_current(psi_plus, psi_0, psi_minus, dpsiP_dx, dpsiP_dy, dpsi0_dx, dpsi0_dy,
                                                         dpsiM_dx, dpsiM_dy)
    mass_cur_x_dx, mass_cur_x_dy = diag.spectral_derivative(mass_cur_x, Kx, Ky, np.fft.fft2, np.fft.ifft2)
    mass_cur_y_dx, mass_cur_y_dy = diag.spectral_derivative(mass_cur_y, Kx, Ky, np.fft.fft2, np.fft.ifft2)

    # curl_of_mass_cur = (mass_cur_y_dx - mass_cur_x_dy).real
    div_mass_cur = (mass_cur_x_dx + mass_cur_y_dy).real

    # Calculate spin current:
    nvz_x = (1 / 2j * (conj(psi_plus) * dpsiP_dx - conj(dpsiP_dx) * psi_plus
                       - (conj(psi_minus) * dpsiM_dx - conj(dpsiM_dx) * psi_minus))).real
    nvz_y = (1 / 2j * (conj(psi_plus) * dpsiP_dy - conj(dpsiP_dy) * psi_plus
                       - (conj(psi_minus) * dpsiM_dy - conj(dpsiM_dy) * psi_minus))).real

    # Calculate the curl:
    nvz_x_dx, nvz_x_dy = diag.spectral_derivative(nvz_x, Kx, Ky, np.fft.fft2, np.fft.ifft2)
    nvz_y_dx, nvz_y_dy = diag.spectral_derivative(nvz_y, Kx, Ky, np.fft.fft2, np.fft.ifft2)

    # curl_spin_z = (nvz_y_dx - nvz_x_dy).real
    div_spin_z = (nvz_x_dx + nvz_y_dy).real

    # Plot curl of spin current
    ax[0].contourf(X, Y, div_spin_z, div_spin_cvals, cmap='BrBG')

    r"""
    for axis in ax:
        if len(psqv) != 0:
            axis.plot(*zip(*psqv), 'bo', markersize=5, label=r'$\sigma_{SQV} = 1$')  # Plots positive SQVs
        if len(nsqv) != 0:
            axis.plot(*zip(*nsqv), 'ro', markersize=5, label=r'$\sigma_{SQV} = -1$')  # Plots negative SQVs
        if len(phqv_plus) != 0:
            axis.plot(*zip(*phqv_plus), 'bX', markersize=5, label=r'$\sigma_1 = 1$')  # Positive HQV in psi_plus
        if len(nhqv_plus) != 0:
            axis.plot(*zip(*nhqv_plus), 'rX', markersize=5, label=r'$\sigma_1 = -1$')  # Negative HQV in psi_plus
        if len(phqv_minus) != 0:
            axis.plot(*zip(*phqv_minus), 'b^', markersize=5, label=r'$\sigma_{-1} = 1$')  # Positive HQV in psi_minus
        if len(nhqv_minus) != 0:
            axis.plot(*zip(*nhqv_minus), 'r^', markersize=5, label=r'$\sigma_{-1} = -1$')  # Negative HQV in psi_minus
        axis.legend()
    """

    # Plot density:
    ax[1].contourf(X, Y, abs(psi_plus) ** 2 + abs(psi_minus) ** 2, levels=25, cmap='gnuplot')

    # Plot curl of mass current:
    ax[2].contourf(X, Y, div_mass_cur, div_mass_cvals, cmap='BrBG')
    plt.suptitle(r'$\tau = {}$'.format(i))

    print('On div frame {}'.format(i))
    cont = [ax[0], ax[1], ax[2]]


# Load in data:
filename = input('Enter data filename: ')
data_file = h5py.File('../../data/{}.hdf5'.format(filename), 'r')

# Other variables:
x, y = data_file['grid/x'][...], data_file['grid/y'][...]
Nx, Ny = len(x), len(y)
dx, dy = x[1] - x[0], y[1] - y[0]
X, Y = np.meshgrid(x[:], y[:])
dkx, dky = 2 * np.pi / (Nx * dx), 2 * np.pi / (Ny * dy)
kxx = np.arange(-Nx // 2, Nx // 2) * dkx
kyy = np.arange(-Nx // 2, Nx // 2) * dky
Kx, Ky = np.meshgrid(kxx, kyy)

r"""
saved_times = data_file['saved_times']
list_of_times = []
for i, time in enumerate(saved_times):
    list_of_times.append([i, time])
print(tabulate(list_of_times, headers=["Frame #", "Time"], tablefmt="orgtbl"))
frame = int(input('Enter the frame number you would like to plot: '))
"""
frame = 0

# ---------------------------------------------------------------------------------------------------------------------
# Curl plots
# ---------------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(20, 10))
for axis in ax:
    axis.set_xlim(x.min(), x.max())
    axis.set_ylim(y.min(), y.max())
    axis.set_xlabel(r'$x/\xi$')
    axis.set_ylabel(r'$y/\xi$')
    axis.set_aspect('equal')

# Load in wavefunction:
psi_plus = data_file['wavefunction/psi_plus'][:, :, frame]
psi_0 = data_file['wavefunction/psi_0'][:, :, frame]
psi_minus = data_file['wavefunction/psi_minus'][:, :, frame]
theta = (np.angle(psi_plus) + np.angle(psi_minus)) / 2

# ---------------------------------------------------------------------------------------------------------------------
# Calculate curl and divergence of spin current:
# ---------------------------------------------------------------------------------------------------------------------
# Get derivatives of wavefunction:
print('Calculating spectral derivative...')
dpsiP_dx, dpsiP_dy = diag.spectral_derivative(psi_plus, Kx, Ky, np.fft.fft2, np.fft.ifft2)
dpsi0_dx, dpsi0_dy = diag.spectral_derivative(psi_0, Kx, Ky, np.fft.fft2, np.fft.ifft2)
dpsiM_dx, dpsiM_dy = diag.spectral_derivative(psi_minus, Kx, Ky, np.fft.fft2, np.fft.ifft2)

# Calculate mass current:
mass_cur_x, mass_cur_y = diag.calculate_mass_current(psi_plus, psi_0, psi_minus, dpsiP_dx, dpsiP_dy, dpsi0_dx, dpsi0_dy,
                                                     dpsiM_dx, dpsiM_dy)
mass_cur_x_dx, mass_cur_x_dy = diag.spectral_derivative(mass_cur_x, Kx, Ky, np.fft.fft2, np.fft.ifft2)
mass_cur_y_dx, mass_cur_y_dy = diag.spectral_derivative(mass_cur_y, Kx, Ky, np.fft.fft2, np.fft.ifft2)

curl_of_mass_cur = (mass_cur_y_dx - mass_cur_x_dy).real
div_mass_cur = (mass_cur_x_dx + mass_cur_y_dy).real

# Calculate spin current:
print('Calculating spin current...')
nvz_x = (1 / 2j * (conj(psi_plus) * dpsiP_dx - conj(dpsiP_dx) * psi_plus
                   - (conj(psi_minus) * dpsiM_dx - conj(dpsiM_dx) * psi_minus))).real
nvz_y = (1 / 2j * (conj(psi_plus) * dpsiP_dy - conj(dpsiP_dy) * psi_plus
                   - (conj(psi_minus) * dpsiM_dy - conj(dpsiM_dy) * psi_minus))).real

# Calculate the curl:
print('Calculating curl of spin current...')
nvz_x_dx, nvz_x_dy = diag.spectral_derivative(nvz_x, Kx, Ky, np.fft.fft2, np.fft.ifft2)
nvz_y_dx, nvz_y_dy = diag.spectral_derivative(nvz_y, Kx, Ky, np.fft.fft2, np.fft.ifft2)

curl_spin_z = (nvz_y_dx - nvz_x_dy).real
div_spin_z = (nvz_x_dx + nvz_y_dy).real
curl_max = np.max([abs(curl_spin_z), abs(curl_of_mass_cur)])

# Plot curl of spin current
curl_spin_cvals = np.linspace(-curl_max, curl_max, 26, endpoint=True)
curl_spin_plot = ax[0].contourf(X, Y, curl_spin_z, curl_spin_cvals, cmap='BrBG')
plt.colorbar(curl_spin_plot, ax=ax[0], fraction=0.045)
ax[0].set_title(r'$\nabla \times n\vec{v}_z^{(spin)}$')

r"""
for axis in ax:
    if len(psqv) != 0:
        axis.plot(*zip(*psqv), 'bo', markersize=5, label=r'$\sigma_{SQV} = 1$')  # Plots positive SQVs
    if len(nsqv) != 0:
        axis.plot(*zip(*nsqv), 'ro', markersize=5, label=r'$\sigma_{SQV} = -1$')  # Plots negative SQVs
    if len(phqv_plus) != 0:
        axis.plot(*zip(*phqv_plus), 'bX', markersize=5, label=r'$\sigma_1 = 1$')  # Positive HQV in psi_plus
    if len(nhqv_plus) != 0:
        axis.plot(*zip(*nhqv_plus), 'rX', markersize=5, label=r'$\sigma_1 = -1$')  # Negative HQV in psi_plus
    if len(phqv_minus) != 0:
        axis.plot(*zip(*phqv_minus), 'b^', markersize=5, label=r'$\sigma_{-1} = 1$')  # Positive HQV in psi_minus
    if len(nhqv_minus) != 0:
        axis.plot(*zip(*nhqv_minus), 'r^', markersize=5, label=r'$\sigma_{-1} = -1$')  # Negative HQV in psi_minus
    axis.legend()
"""

# Plot density:
dens = abs(psi_plus) ** 2 + abs(psi_0) ** 2 + abs(psi_minus) ** 2
dens_cvals = np.linspace(0, np.max(dens), 26)
dens_plot = ax[1].contourf(X, Y, dens, dens_cvals, cmap='gnuplot')
plt.colorbar(dens_plot, ax=ax[1], fraction=0.045)
ax[1].set_title(r'Overall atomic density')

# Plot curl of mass current:
curl_mass_cvals = np.linspace(-curl_max, curl_max, 26, endpoint=True)
curl_mass_plot = ax[2].contourf(X, Y, curl_of_mass_cur, curl_mass_cvals, cmap='BrBG')
plt.colorbar(curl_mass_plot, ax=ax[2], fraction=0.045)
ax[2].set_title(r'$\nabla \times n\vec{v}^{(mass)}$')

# plt.show()
cont = []   # Global cont used for animations
anim = animation.FuncAnimation(fig, animate_curl, frames=200)
anim.save('./{}_curl.mp4'.format(filename), dpi=200, writer=animation.FFMpegWriter(fps=30))

# ---------------------------------------------------------------------------------------------------------------------
# Divergence plots
# ---------------------------------------------------------------------------------------------------------------------
# Divergence plots:
div_max = np.max([abs(div_spin_z), abs(div_mass_cur)])

fig, ax = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(20, 10))
for axis in ax:
    axis.set_xlim(x.min(), x.max())
    axis.set_ylim(y.min(), y.max())
    axis.set_xlabel(r'$x/\xi$')
    axis.set_ylabel(r'$y/\xi$')
    axis.set_aspect('equal')

# Plot curl of spin current
div_spin_cvals = np.linspace(-div_max, div_max, 26, endpoint=True)
div_spin_plot = ax[0].contourf(X, Y, div_spin_z, div_spin_cvals, cmap='BrBG')
plt.colorbar(div_spin_plot, ax=ax[0], fraction=0.045)
ax[0].set_title(r'$\nabla \cdot n\vec{v}_z^{(spin)}$')

r"""
for axis in ax:
    if len(psqv) != 0:
        axis.plot(*zip(*psqv), 'bo', markersize=5, label=r'$\sigma_{SQV} = 1$')  # Plots positive SQVs
    if len(nsqv) != 0:
        axis.plot(*zip(*nsqv), 'ro', markersize=5, label=r'$\sigma_{SQV} = -1$')  # Plots negative SQVs
    if len(phqv_plus) != 0:
        axis.plot(*zip(*phqv_plus), 'bX', markersize=5, label=r'$\sigma_1 = 1$')  # Positive HQV in psi_plus
    if len(nhqv_plus) != 0:
        axis.plot(*zip(*nhqv_plus), 'rX', markersize=5, label=r'$\sigma_1 = -1$')  # Negative HQV in psi_plus
    if len(phqv_minus) != 0:
        axis.plot(*zip(*phqv_minus), 'b^', markersize=5, label=r'$\sigma_{-1} = 1$')  # Positive HQV in psi_minus
    if len(nhqv_minus) != 0:
        axis.plot(*zip(*nhqv_minus), 'r^', markersize=5, label=r'$\sigma_{-1} = -1$')  # Negative HQV in psi_minus
    axis.legend()
"""

# Plot density:
dens_plot = ax[1].contourf(X, Y, dens, dens_cvals, cmap='gnuplot')
plt.colorbar(dens_plot, ax=ax[1], fraction=0.045)
ax[1].set_title(r'Overall atomic density')

# Plot curl of mass current:
div_mass_cvals = np.linspace(-div_max, div_max, 26, endpoint=True)
div_mass_plot = ax[2].contourf(X, Y, div_mass_cur, div_mass_cvals, cmap='BrBG')
plt.colorbar(div_mass_plot, ax=ax[2], fraction=0.045)
ax[2].set_title(r'$\nabla \cdot n\vec{v}^{(mass)}$')

cont = []   # Global cont used for animations
anim = animation.FuncAnimation(fig, animate_div, frames=200)
anim.save('./{}_div.mp4'.format(filename), dpi=200, writer=animation.FFMpegWriter(fps=30))
# plt.show()
