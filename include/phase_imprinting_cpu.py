import numpy as np


def get_phase_cpu(num_of_vort, pos, grid_x, grid_y):
    """
    num_of_vort: number of vortices to imprint
    pos: iterable of positions to imprint the vortices
    grid_x: X-meshgrid
    grid_y: Y-meshgrid
    """

    # Constructing necessary grid parameters:
    x_pts, y_pts = len(grid_x[:, 0]), len(grid_y[0, :])
    dx, dy = grid_x[0, 1] - grid_x[0, 0], grid_y[1, 0] - grid_y[0, 0]
    grid_len_x, grid_len_y = x_pts * dx, y_pts * dy

    # Phase initialisation
    theta_tot = np.empty((x_pts, y_pts))

    # Scale pts:
    x_tilde = 2 * np.pi * ((grid_x - grid_x.min()) / grid_len_x)
    y_tilde = 2 * np.pi * ((grid_y - grid_y.min()) / grid_len_y)

    for i in range(num_of_vort // 2):
        print(f'On iteration {i}')
        theta_k = np.zeros((x_pts, y_pts))

        x_m, y_m = next(pos)
        x_p, y_p = next(pos)

        # Scaling vortex positions:
        x_m_tilde = 2 * np.pi * ((x_m - grid_x.min()) / grid_len_x)
        y_m_tilde = 2 * np.pi * ((y_m - grid_y.min()) / grid_len_y)
        x_p_tilde = 2 * np.pi * ((x_p - grid_x.min()) / grid_len_x)
        y_p_tilde = 2 * np.pi * ((y_p - grid_y.min()) / grid_len_y)

        # Aux variables
        Y_minus = y_tilde - y_m_tilde
        X_minus = x_tilde - x_m_tilde
        Y_plus = y_tilde - y_p_tilde
        X_plus = x_tilde - x_p_tilde

        heav_xp = np.heaviside(X_plus, 1.)
        heav_xm = np.heaviside(X_minus, 1.)

        for nn in np.arange(-5, 6):
            theta_k += np.arctan(np.tanh((Y_minus + 2 * np.pi * nn) / 2) * np.tan((X_minus - np.pi) / 2)) \
                                - np.arctan(np.tanh((Y_plus + 2 * np.pi * nn) / 2) * np.tan((X_plus - np.pi) / 2)) \
                                + np.pi * (heav_xp - heav_xm)

        theta_k -= y_tilde * (x_p_tilde - x_m_tilde) / (2 * np.pi)
        theta_tot += theta_k

    return theta_tot


def get_positions_cpu(num_of_vortices, threshold, len_x, len_y):
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
        if np.mod(len(accepted_pos), 500) == 0:
            print('Found {} positions...'.format(len(accepted_pos)))

    print('Found {} positions in {} iterations.'.format(len(accepted_pos), iterations))
    return iter(accepted_pos)
