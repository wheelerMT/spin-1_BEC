import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
import numexpr as ne


def makeSpinMagExpectationVideo(filename):
    global cont
    # ----------------- Initialising contours and cvals for spin plots ---------------------------
    fig_spin, axes_spin = plt.subplots(1, 3, sharey=True, figsize=(10, 6))

    for axis in axes_spin:
        axis.set_aspect('equal')

    dens_max = np.max([abs(psi_plus[:, :, -1]) ** 2, abs(psi_0[:, :, -1]) ** 2, abs(psi_minus[:, :, -1]) ** 2])
    cvals_spin = np.linspace(0, 1, num=25, endpoint=True)  # Spin contour values
    cvals_dens = np.linspace(0, dens_max, num=25, endpoint=True)  # Contour values

    # Initialising contours
    cont_splus = axes_spin[0].contourf(X, Y, abs(psi_plus[:, :, 0]) ** 2, cvals_dens, cmap='gnuplot')
    cont_spin = axes_spin[1].contourf(X, Y, spin_expec_mag[:, :, 0], cvals_spin, cmap='PuRd')
    cont_sminus = axes_spin[2].contourf(X, Y, abs(psi_minus[:, :, 0]) ** 2, cvals_dens, cmap='gnuplot')
    cont = [cont_splus, cont_spin, cont_sminus]

    # Density colorbar
    divider = make_axes_locatable(axes_spin[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar_dens = fig_spin.colorbar(cont_sminus, cax=cax, orientation='vertical')
    cbar_dens.formatter.set_powerlimits((0, 0))
    cbar_dens.update_ticks()

    # Spin expec mag colorbar
    divider = make_axes_locatable(axes_spin[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar_spin = fig_spin.colorbar(cont_spin, cax=cax, orientation='vertical', ticks=[0, 1])
    cbar_spin.ax.set_yticklabels = ['0', '1']

    fig_spin.tight_layout(rect=[0.02, 0.03, 1, 0.75])
    # -----------------------------------------------------------------------------------------------

    # Sets axes limits and titles
    for ax in axes_spin:
        ax.set_xlim(np.min(x[:]), np.max(x[:]))
        ax.set_ylim(np.min(y[:]), np.max(y[:]))
        ax.set_xlabel(r'$x / \xi$')

        if ax == axes_spin[0]:
            ax.set_title(r'$|\psi_+|^2$')
            ax.set_ylabel(r'$y / \xi$')
        if ax == axes_spin[2]:
            ax.set_title(r'$|\psi_-|^2$')
        if ax == axes_spin[1]:
            ax.set_aspect('equal')
            ax.set_title(r'$|<F>|$')

    # Animation function
    def animate_spin(i):

        global cont
        for contour in cont:
            for c in contour.collections:
                c.remove()

        axes_spin[0].contourf(X, Y, abs(psi_plus[:, :, i]) ** 2, cvals_dens, cmap='gnuplot')
        axes_spin[1].contourf(X, Y, spin_expec_mag[:, :, i], cvals_spin, cmap='PuRd')
        axes_spin[2].contourf(X, Y, abs(psi_minus[:, :, i]) ** 2, cvals_dens, cmap='gnuplot')

        cont = [axes_spin[0], axes_spin[1], axes_spin[2]]
        print('On spin iteration %i' % (i + 1))
        plt.suptitle(r'$\tau$ = %2f' % (Nframe * dt * i), y=0.8)
        return cont

    # Calls the animation function and saves the result
    anim = animation.FuncAnimation(fig_spin, animate_spin, frames=num_of_frames, repeat=False)
    anim.save('../../images/unsorted/{}'.format(filename), dpi=200,
              writer=animation.FFMpegWriter(fps=60, codec="libx264", extra_args=['-pix_fmt', 'yuv420p']))
    print('Spin video saved successfully.')


# ---------------------------------------------------------------------------------------------------------------------
# Loading data
# ---------------------------------------------------------------------------------------------------------------------
filename = input('Enter filename: ')
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

# Magnitude of spin expectation
print('Calculating spin vectors...')
spin_expec_mag = np.empty((Nx, Ny, num_of_frames), dtype='float32')

for i in range(num_of_frames):
    psi_p2d = psi_plus[:, :, i]
    psi_02d = psi_0[:, :, i]
    psi_m2d = psi_minus[:, :, i]
    n_2d = abs(psi_p2d) ** 2 + abs(psi_02d) ** 2 + abs(psi_m2d) ** 2

    Fx = ne.evaluate("1 / sqrt(2) * (conj(psi_p2d) * psi_02d + conj(psi_02d) * (psi_p2d + psi_m2d) "
                     "+ conj(psi_m2d) * psi_02d)").real
    Fy = ne.evaluate("1j / sqrt(2) * (-conj(psi_p2d) * psi_02d + conj(psi_02d) * (psi_p2d - psi_m2d) "
                     "+ conj(psi_m2d) * psi_02d)").real
    Fz = ne.evaluate("abs(psi_p2d).real ** 2 - abs(psi_m2d).real ** 2")

    spin_expec_mag[:, :, i] = ne.evaluate("sqrt(Fx ** 2 + Fy ** 2 + Fz ** 2) / n_2d")

    if i % 10 == 0:
        print('Calculating spin vectors: %i%% complete. ' % (i / num_of_frames * 100))

# ---------------------------------------------------------------------------------------------------------------------
# Making videos
# ---------------------------------------------------------------------------------------------------------------------
makeSpinMagExpectationVideo(filename='{}_vid.mp4'.format(filename))  # Creates spin magnitude video
data_file.close()
