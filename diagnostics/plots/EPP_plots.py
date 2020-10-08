import h5py
import matplotlib.pyplot as plt
import numpy as np
import include.vortex_detection as vd
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from scipy.signal import savgol_filter


def generate_vortex_lists(wfn_plus, wfn_minus, grid_x, grid_y):
    """ Returns lists of vortex positions at each frame. """
    # Do vortex detection:
    vortex_counter = []
    psqv_frames, nsqv_frames = [], []
    phqv_plus_frames, nhqv_plus_frames = [], []
    phqv_minus_frames, nhqv_minus_frames = [], []
    for i in range(59, 500):
        print('On frame {}:'.format(i))

        psqv, nsqv, phqv_plus, nhqv_plus, phqv_minus, nhqv_minus, num_of_vort = \
            vd.calculate_vortices(wfn_plus[:, :, i], wfn_minus[:, :, i], grid_x, grid_y)
        psqv_frames.append(psqv), nsqv_frames.append(nsqv)
        phqv_plus_frames.append(phqv_plus), nhqv_plus_frames.append(nhqv_plus)
        phqv_minus_frames.append(phqv_minus), nhqv_minus_frames.append(nhqv_minus)
        vortex_counter.append(num_of_vort)

    return psqv_frames, nsqv_frames, phqv_plus_frames, nhqv_plus_frames, phqv_minus_frames, nhqv_minus_frames, vortex_counter


def makeDensityVideo(filename):
    global cont_dens
    # ----------------- Initialising contours and cvals for density plots ---------------------------
    fig_dens, axes_dens = plt.subplots(1, 2, sharey=True, figsize=(8, 8))

    for axis in axes_dens:
        axis.set_aspect('equal')

    dens_max = 2500
    cvals_dens = np.linspace(0, dens_max, num=50, endpoint=True)  # Contour values

    # Initialising contours and colour bar
    cont_plus = axes_dens[0].contourf(X, Y, abs(psi_plus[:, :, 0]) ** 2, cvals_dens, cmap='gnuplot')
    cont_minus = axes_dens[1].contourf(X, Y, abs(psi_minus[:, :, 0]) ** 2, cvals_dens, cmap='gnuplot')
    cont_dens = [cont_plus, cont_minus]

    divider = make_axes_locatable(axes_dens[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cont_plus, cax=cax)

    # -----------------------------------------------------------------------------------------------

    # Sets axes limits and titles
    for ax in axes_dens:
        ax.set_xlim(np.min(x[:]), np.max(x[:]))
        ax.set_ylim(np.min(y[:]), np.max(y[:]))
        ax.set_xlabel(r'$x / \xi$')

        if ax == axes_dens[0]:
            ax.set_title(r'$|\psi_+|^2$')
            ax.set_ylabel(r'$y / \xi $')
        elif ax == axes_dens[1]:
            ax.set_title(r'$|\psi_-|^2$')

    # Animation function
    def animate(i):
        t1 = time.time()
        global cont_dens, points_list
        global psqv_pts, nsqv_pts

        for contour in cont_dens:
            for c in contour.collections:
                c.remove()
        if i == 0:
            points_list = []
        for pts in points_list:
            pts.remove()

        axes_dens[0].contourf(X, Y, abs(psi_plus[:, :, i]) ** 2, cvals_dens, cmap='gnuplot')
        axes_dens[1].contourf(X, Y, abs(psi_minus[:, :, i]) ** 2, cvals_dens, cmap='gnuplot')

        for axis in axes_dens:
            psqv_pts, = axis.plot(*zip(*psqv_list[i]), 'wo', markersize=7,
                                  label=r'$\sigma_{SQV} = 1$')  # Plots positive SQVs
            nsqv_pts, = axis.plot(*zip(*nsqv_list[i]), 'ko', markersize=7,
                                  label=r'$\sigma_{SQV} = -1$')  # Plots negative SQVs
            if axis == axes_dens[0]:
                phqv_plus_pts, = axis.plot(*zip(*phqv_plus_list[i]), 'wX', markersize=7,
                                           label=r'$\sigma_1 = 1$')  # Positive HQV in psi_plus

                nhqv_plus_pts, = axis.plot(*zip(*nhqv_plus_list[i]), 'kX', markersize=7,
                                           label=r'$\sigma_1 = -1$')  # Negative HQV in psi_plus
            if axis == axes_dens[1]:
                phqv_minus_pts, = axis.plot(*zip(*phqv_minus_list[i]), 'w^', markersize=7,
                                            label=r'$\sigma_{-1} = 1$')  # Positive HQV in psi_minus

                nhqv_minus_pts, = axis.plot(*zip(*nhqv_minus_list[i]), 'k^', markersize=7,
                                            label=r'$\sigma_{-1} = -1$')  # Negative HQV in psi_minus
        points_list.append(psqv_pts)
        points_list.append(nsqv_pts)
        cont_dens = [axes_dens[0], axes_dens[1]]
        print('On density iteration %i' % (i + 1))
        plt.suptitle(r'$\tau$ = %2f' % (Nframe * dt * i), y=0.7)
        print('Plot {} took {}s.'.format(i, time.time() - t1))

        return cont_dens

    # Calls the animation function and saves the result
    anim = animation.FuncAnimation(fig_dens, animate, frames=num_of_frames // 2, repeat=False)
    anim.save('video.mp4', dpi=200, writer=animation.FFMpegWriter(fps=10))
    print('Density video saved successfully.')


def vortex_number_plots(vortices, t):
    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=int)
        ret[n:] = ret[n:] - ret[:-n]
        return int(ret[n - 1:] / n)

    vortices = moving_average(vortices)
    vortices = savgol_filter(vortices, 51, 1)

    # Generate figure:
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.set_xlabel(r'$t/\tau$')
    ax.set_ylabel(r'$N_{vort}$')
    ax.loglog(t, vortices)
    plt.savefig('../../images/vortex_number.png', dpi=200)


# Load in data file:
filename = '1024nomag'  # input('Enter name of data file: ')
data_file = h5py.File('../../../scratch/data/polar_phase/{}.hdf5'.format(filename), 'r')

# --------------------------------------------------------------------------------------------------------------------
# Spatial and Potential parameters:
# --------------------------------------------------------------------------------------------------------------------
x = data_file['grid/x'][...]
y = data_file['grid/x'][...]
X, Y = np.meshgrid(x, y)  # Spatial meshgrid

# Wavefunction data
psi_plus = data_file['wavefunction/psi_plus']
psi_minus = data_file['wavefunction/psi_minus']
num_of_frames = psi_plus.shape[-1]

# Loading time variables:
dt, Nframe = 1e-2, 10000
time_array = np.arange(6000, 50000, 100)

print('Generating vortex lists...')
psqv_list, nsqv_list, phqv_plus_list, nhqv_plus_list, phqv_minus_list, nhqv_minus_list, vortex_number \
    = generate_vortex_lists(psi_plus, psi_minus, x, y)

vortex_number_plots(vortex_number, time_array)

r"""
# Plot overlay of vortices:
frame = 6
ax[0].contourf(X, Y, abs(psi_plus[:, :, frame]) ** 2, levels=50, cmap='gnuplot')
ax1 = ax[1].contourf(X, Y, abs(psi_minus[:, :, frame]) ** 2, levels=50, cmap='gnuplot')

for axis in ax:
    if len(psqv_list[frame]) != 0:
        axis.plot(*zip(*psqv_list[frame]), 'wo', markersize=7, label=r'$\sigma_{SQV} = 1$')  # Plots positive SQVs
    if len(nsqv_list[frame]) != 0:
        axis.plot(*zip(*nsqv_list[frame]), 'ko', markersize=7, label=r'$\sigma_{SQV} = -1$')  # Plots negative SQVs
    if axis == ax[0]:
        if len(phqv_plus_list[frame]) != 0:
            axis.plot(*zip(*phqv_plus_list[frame]), 'wX', markersize=7, label=r'$\sigma_1 = 1$')  # Positive HQV in psi_plus
        if len(nhqv_plus_list[frame]) != 0:
            axis.plot(*zip(*nhqv_plus_list[frame]), 'kX', markersize=7, label=r'$\sigma_1 = -1$')  # Negative HQV in psi_plus
    if axis == ax[1]:
        if len(phqv_minus_list[frame]) != 0:
            axis.plot(*zip(*phqv_minus_list[frame]), 'w^', markersize=7, label=r'$\sigma_{-1} = 1$')  # Positive HQV in psi_minus
        if len(nhqv_minus_list[frame]) != 0:
            axis.plot(*zip(*nhqv_minus_list[frame]), 'k^', markersize=7, label=r'$\sigma_{-1} = -1$')  # Negative HQV in psi_minus

plt.colorbar(ax1, ax = ax[1])
plt.show()
"""
