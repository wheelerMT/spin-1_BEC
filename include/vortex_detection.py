import numpy as np
import time


def calculate_vortices(wfn_plus, wfn_minus, grid_x, grid_y):
    def vortex_detection_unwrapped(pos_x, pos_y, wfn, x_grid, y_grid):
        """A plaquette detection algorithm that finds areas of 2pi winding and then performs a least squares
        fit in order to obtain exact position of the vortex core."""

        # Unwrap phase to avoid discontinuities:
        phase_x = np.unwrap(np.angle(wfn), axis=0)
        phase_y = np.unwrap(np.angle(wfn), axis=1)

        # Empty list for storing positions:
        vortex_pos = []
        antivortex_pos = []

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
            if np.round(phase_sum, 4) == np.round(2 * np.pi, 4):
                antivortex_pos.append(refine_positions([ii, jj], wfn, x_grid, y_grid))  # Vortex
            elif np.round(phase_sum, 4) == np.round(-2 * np.pi, 4):
                vortex_pos.append(refine_positions([ii, jj], wfn, x_grid, y_grid))  # Anti-vortex

        return vortex_pos, antivortex_pos

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

    def remove_invalid_positions(vortex_pos, antivortex_pos, r):
        """ Function that removes oppositely signed vortices that are too close together, typically these are sometimes
        detected by the algorithm when the phase field is broken near an existing vortex."""

        removable_vort = []  # List of removable vortices
        removable_antivort = []  # List of removable anti-vortices

        for vtx in vortex_pos:
            position_found = False

            for anti_vtx in antivortex_pos:
                if (vtx[0] - anti_vtx[0]) ** 2 + (vtx[1] - anti_vtx[1]) ** 2 < r:
                    if anti_vtx not in removable_antivort:
                        removable_antivort.append(anti_vtx)
                        position_found = True

            if position_found and vtx not in removable_vort:
                removable_vort.append(vtx)

        for vtx in removable_vort:
            if vtx in vortex_pos:
                del vortex_pos[vortex_pos.index(vtx)]

        for anti_vtx in removable_antivort:
            if anti_vtx in antivortex_pos:
                del antivortex_pos[antivortex_pos.index(anti_vtx)]

        return vortex_pos, antivortex_pos

    # Calculate grid properties:
    Nx, Ny = len(grid_x), len(grid_y)
    dx, dy = grid_x[1] - grid_x[0], grid_y[1] - grid_y[0]
    n_inf = np.sum(abs(wfn_plus) ** 2 + abs(wfn_minus) ** 2) / (dx * dy * Nx * Ny)

    # Finds indices where density falls below threshold value:
    eps = 0.1  # Threshold percentage
    dens_plus_x, dens_plus_y = np.where(abs(wfn_plus) ** 2 < eps * n_inf)
    dens_minus_x, dens_minus_y = np.where(abs(wfn_minus) ** 2 < eps * n_inf)

    # Plaquette vortex detection:
    t1 = time.time()
    # Find where phase winds by 2pi:
    vortex_plus, antivortex_plus = vortex_detection_unwrapped(dens_plus_x, dens_plus_y, wfn_plus, grid_x, grid_y)
    vortex_minus, antivortex_minus = vortex_detection_unwrapped(dens_minus_x, dens_minus_y, wfn_minus, grid_x, grid_y)

    # Performs analysis to determine type of vortex:
    # Find SQVs by finding where the positive/negative vortices overlap in the + and - component:
    psqv = []
    nsqv = []
    radius = 4

    for pos_plus in vortex_plus:
        for pos_minus in vortex_minus:
            if (pos_plus[0] - pos_minus[0]) ** 2 + (pos_plus[1] - pos_minus[1]) ** 2 < radius:
                psqv.append(pos_plus)
                break

    for pos_plus in antivortex_plus:
        for pos_minus in antivortex_minus:
            if (pos_plus[0] - pos_minus[0]) ** 2 + (pos_plus[1] - pos_minus[1]) ** 2 < radius:
                nsqv.append(pos_plus)
                break

    # Remove SQV positions from master vortex list:
    for vortex in psqv:
        if vortex in vortex_plus:
            del vortex_plus[vortex_plus.index(vortex)]
        if vortex in vortex_minus:
            del vortex_minus[vortex_minus.index(vortex)]
    for antivortex in nsqv:
        if antivortex in antivortex_plus:
            del antivortex_plus[antivortex_plus.index(antivortex)]
        if antivortex in antivortex_minus:
            del antivortex_minus[antivortex_minus.index(antivortex)]

    # Remove opposite signed SQVs too close together:
    psqv, nsqv = remove_invalid_positions(psqv, nsqv, 4)

    # HQV positions are vortices whose cores do not overlap:
    phqv_plus = []
    for pos_plus in vortex_plus:
        near_pt = False
        for pos in psqv:
            if (pos_plus[0] - pos[0]) ** 2 + (pos_plus[1] - pos[1]) ** 2 < radius:
                near_pt = True
        if near_pt is False:
            phqv_plus.append(pos_plus)

    nhqv_plus = []
    for pos_plus in antivortex_plus:
        near_pt = False
        for pos in nsqv:
            if (pos_plus[0] - pos[0]) ** 2 + (pos_plus[1] - pos[1]) ** 2 < radius:
                near_pt = True
        if near_pt is False:
            nhqv_plus.append(pos_plus)

    phqv_minus = []
    for pos_minus in vortex_minus:
        near_pt = False
        for pos in psqv:
            if (pos_minus[0] - pos[0]) ** 2 + (pos_minus[1] - pos[1]) ** 2 < radius:
                near_pt = True
        if near_pt is False:
            phqv_minus.append(pos_minus)

    nhqv_minus = []
    for pos_minus in antivortex_minus:
        near_pt = False
        for pos in nsqv:
            if (pos_minus[0] - pos[0]) ** 2 + (pos_minus[1] - pos[1]) ** 2 < radius:
                near_pt = True
        if near_pt is False:
            nhqv_minus.append(pos_minus)

    # Remove opposite signed HQVs too close together:
    phqv_plus, nhqv_plus = remove_invalid_positions(phqv_plus, nhqv_plus, 4)
    phqv_minus, nhqv_minus = remove_invalid_positions(phqv_minus, nhqv_minus, 4)

    num_of_vortices = len(psqv) + len(nsqv) + len(phqv_plus) + len(nhqv_plus) + len(phqv_minus) + len(nhqv_minus)
    print('Detected a total of {:d} vortices in {:.3f}s!'.format(num_of_vortices, time.time() - t1))

    return psqv, nsqv, phqv_plus, nhqv_plus, phqv_minus, nhqv_minus, num_of_vortices
