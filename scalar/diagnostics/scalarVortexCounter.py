import h5py
import numpy as np


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


# Import dataset:
data = h5py.File('../../data/frames/100kf_scalar.hdf5', 'r')

# Load in data:
x, y = data['grid/x'][...], data['grid/y'][...]
X, Y = np.meshgrid(x, y)
psi = data['wavefunction/psi']

vortex_number = []
num_of_frames = psi.shape[-1]

# Detect and count vortices:
dt = 1e-2
Nframe = 10000
time_to_start = 10000   # Time in the simulation to start counting
start_frame = int(time_to_start // (dt * Nframe))
time_array = [time_to_start + i * dt for i in range(num_of_frames - start_frame)]

# Calculate grid properties:
Nx, Ny = len(x), len(y)
dx, dy = x[1] - x[0], y[1] - y[0]
eps = 0.1  # Threshold percentage

# Loops through data to find vortices:
for i in range(start_frame, psi.shape[-1]):
    # Finds indices where density falls below threshold value:
    n_inf = np.sum(abs(psi[:, :, i]) ** 2) / (dx * dy * Nx * Ny)
    dens_x, dens_y = np.where(abs(psi[:, :, i]) ** 2 < eps * n_inf)

    vortex_pos, antivortex_pos = vortex_detection_unwrapped(dens_x, dens_y, psi[:, :, i], x, y)
    vortex_number.append(len(vortex_pos + antivortex_pos))

# Generate vortex file for saving data to:
vortex_file = h5py.File('scalar_vortex_data.hdf5', 'w')
vortex_file.create_dataset('vortex_number', data=vortex_number)
