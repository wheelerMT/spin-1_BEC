import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
import h5py
import time


def vortex_detection_unwrapped(pos_x, pos_y, wfn, x_grid, y_grid):
    """A plaquette detection algorithm that finds areas of 2pi winding and then performs a least squares
    fit in order to obtain exact position of the vortex core."""

    # Unwrap phase to avoid discontinuities:
    phase_x = np.unwrap(np.angle(wfn), axis=0)
    phase_y = np.unwrap(np.angle(wfn), axis=1)

    # Empty list for storing positions:
    vort_pos = []
    antivort_pos = []

    # Sum phase difference over plaquettes:
    for ii, jj in zip(pos_x, pos_y):
        # Ensures algorithm works at edge of box:
        if ii == len(x_grid) - 1:
            ii = -1
        if jj == len(y_grid) - 1:
            jj = -1

        phase_sum = 0
        phase_sum += phase_x[ii, jj + 1] - phase_x[ii, jj]
        phase_sum += phase_y[ii + 1, jj + 1] - phase_y[ii, jj + 1]
        phase_sum += phase_x[ii + 1, jj] - phase_x[ii + 1, jj + 1]
        phase_sum += phase_y[ii, jj] - phase_y[ii + 1, jj]

        # If sum of phase difference is 2pi or -2pi, take note of vortex position:
        if np.round(abs(phase_sum), 4) == np.round(2 * np.pi, 4):
            if phase_sum > 0:  # Vortex
                antivort_pos.append(refine_positions([ii, jj], wfn, x_grid, y_grid))
            elif phase_sum < 0:  # Anti-vortex
                vort_pos.append(refine_positions([ii, jj], wfn, x_grid, y_grid))

    return vort_pos, antivort_pos


def refine_positions(position, wfn, x_grid, y_grid):
    """ Perform a least squares fitting to correct the vortex positions."""

    x_pos, y_pos = position
    # If at edge of grid, skip the correction:
    if x_pos == len(x_grid) - 1:
        return x_pos, y_pos
    if y_pos == len(y_grid) - 1:
        return x_pos, y_pos

    x_update = (-wfn[x_pos, y_pos] - wfn[x_pos, y_pos + 1] + wfn[x_pos + 1, y_pos] + wfn[x_pos + 1, y_pos + 1]) / 2
    y_update = (-wfn[x_pos, y_pos] + wfn[x_pos, y_pos + 1] - wfn[x_pos + 1, y_pos] + wfn[x_pos + 1, y_pos + 1]) / 2
    c_update = (3 * wfn[x_pos, y_pos] + wfn[x_pos, y_pos + 1] + wfn[x_pos + 1, y_pos] - wfn[
        x_pos + 1, y_pos + 1]) / 4

    Rx, Ry = x_update.real, y_update.real
    Ix, Iy = x_update.imag, y_update.imag
    Rc, Ic = c_update.real, c_update.imag

    det = 1 / (Rx * Iy - Ry * Ix)
    delta_x = det * (Iy * Rc - Ry * Ic)
    delta_y = det * (-Ix * Rc + Rx * Ic)

    x_v = x_pos - delta_x
    y_v = y_pos - delta_y

    # Return x and y positions:
    return (y_v - len(y_grid) // 2) * (y_grid[1] - y_grid[0]), (x_v - len(x_grid) // 2) * (x_grid[1] - x_grid[0])


# Load the data:
filename = 'scalar'
data_path = '../../data/{}.hdf5'.format(filename)

data_file = h5py.File(data_path, 'r')

# Loading grid array data:
x, y = data_file['grid/x'], data_file['grid/y']
X, Y = np.meshgrid(x[:], y[:])
Nx, Ny = x[:].size, y[:].size
dx, dy = x[1] - x[0], y[1] - y[0]

# Load in time data:
Nt, dt, Nframe = data_file['time/Nt'][...], data_file['time/dt'][...], data_file['time/Nframe'][...]

# Load in wavefunction:
psi = data_file['wavefunction/psi']
num_of_frames = psi.shape[-1]

# Determine contour values:
cvals = np.linspace(0, 2000, 50, endpoint=True)
eps = 0.1

# Generate the figure:
fig, ax = plt.subplots(1, )
ax.set_xlabel(r'$x / \xi$')
ax.set_ylabel(r'$y / \xi$')
ax.set_xlim(x[:].min(), x[:].max())
ax.set_ylim(y[:].min(), y[:].max())
colors = ['w', 'k']

# Initial plot:
contour = [ax.contourf(X, Y, abs(psi[:, :, 0]) ** 2, cvals, cmap='gnuplot')]
# Find vortex positions:
n_inf = np.sum(abs(psi[:, :, 0]) ** 2) / (dx * dy * Nx * Ny)
dens_x, dens_y = np.where(abs(psi[:, :, 0]) ** 2 < eps * n_inf)
vortex_pos, antivortex_pos = vortex_detection_unwrapped(dens_x, dens_y, psi[:, :, 0], x, y)
points = [ax.plot(*zip(*p), marker='o', markersize=2, color=colors[j],
                  linestyle='None')[0] for j, p in enumerate([vortex_pos, antivortex_pos])]


# Function to update plot:
def update_plot(i):
    global points
    # Finds indices where density falls below threshold value:
    n_inf = np.sum(abs(psi[:, :, i]) ** 2) / (dx * dy * Nx * Ny)
    dens_x, dens_y = np.where(abs(psi[:, :, i]) ** 2 < eps * n_inf)
    vortex_pos, antivortex_pos = vortex_detection_unwrapped(dens_x, dens_y, psi[:, :, i], x, y)

    t1 = time.time()
    for tp in contour[0].collections:
        tp.remove()
    for p in points:
        p.remove()
    points = [ax.plot(*zip(*p), marker='o', markersize=2, color=colors[j],
                      linestyle='None')[0] for j, p in enumerate([vortex_pos, antivortex_pos])]
    contour[0] = ax.contourf(X, Y, abs(psi[:, :, i]) ** 2, cvals, cmap='gnuplot')

    plt.suptitle(r'$\tau = {:.2f}$'.format(i * dt * Nframe))
    print('On plot {}. The time for this plot was {:.2f}s'.format(i, time.time()-t1))
    return contour[0].collections


anim = matplotlib.animation.FuncAnimation(fig, update_plot, frames=num_of_frames,
                                          interval=10, blit=False, repeat=True)
anim.save('scalar_turbulence.mp4', dpi=400,
          writer=matplotlib.animation.FFMpegWriter(fps=60))
