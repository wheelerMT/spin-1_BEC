import cupy as cp
import numpy as np


def get_phase(num_of_vort, pos, x_pts, y_pts, grid_x, grid_y, grid_len_x, grid_len_y):
    # Phase initialisation
    theta_tot = cp.empty((x_pts, y_pts))

    # Scale pts:
    x_tilde = 2 * cp.pi * ((grid_x - grid_x.min()) / grid_len_x)
    y_tilde = 2 * cp.pi * ((grid_y - grid_y.min()) / grid_len_y)

    for _ in range(num_of_vort // 2):
        theta_k = cp.zeros((x_pts, y_pts))

        x_m, y_m = next(pos)
        x_p, y_p = next(pos)

        # Scaling vortex positions:
        x_m_tilde = 2 * cp.pi * ((x_m - grid_x.min()) / grid_len_x)
        y_m_tilde = 2 * cp.pi * ((y_m - grid_y.min()) / grid_len_y)
        x_p_tilde = 2 * cp.pi * ((x_p - grid_x.min()) / grid_len_x)
        y_p_tilde = 2 * cp.pi * ((y_p - grid_y.min()) / grid_len_y)

        # Aux variables
        Y_minus = y_tilde - y_m_tilde
        X_minus = x_tilde - x_m_tilde
        Y_plus = y_tilde - y_p_tilde
        X_plus = x_tilde - x_p_tilde

        heav_xp = cp.asarray(np.heaviside(cp.asnumpy(X_plus), 1.))
        heav_xm = cp.asarray(np.heaviside(cp.asnumpy(X_minus), 1.))

        for nn in cp.arange(-5, 6):
            theta_k += cp.arctan(cp.tanh((Y_minus + 2 * cp.pi * nn) / 2) * cp.tan((X_minus - cp.pi) / 2)) \
                                - cp.arctan(cp.tanh((Y_plus + 2 * cp.pi * nn) / 2) * cp.tan((X_plus - cp.pi) / 2)) \
                                + cp.pi * (heav_xp - heav_xm)

        theta_k -= y_tilde * (x_p_tilde - x_m_tilde) / (2 * cp.pi)
        theta_tot += theta_k

    return theta_tot


def get_positions(num_of_vortices, threshold, len_x, len_y):
    accepted_pos = []
    iterations = 0
    while len(accepted_pos) < num_of_vortices:
        within_range = True
        while within_range:
            pos = np.random.uniform(-len_x // 2, len_x // 2), np.random.uniform(-len_y // 2, len_y // 2)
            iterations += 1
            triggered = False
            for accepted_position in accepted_pos:
                if abs(pos[0] - accepted_position[0]) < threshold:
                    if abs(pos[1] - accepted_position[1]) < threshold:
                        triggered = True
                        break
            if not triggered:
                accepted_pos.append(pos)
                within_range = False
        if np.mod(len(accepted_pos), 10) == 0:
            print('Found {} positions...'.format(len(accepted_pos)))

    print('Found {} positions in {} iterations.'.format(len(accepted_pos), iterations))
    return iter(accepted_pos)
