import numpy as np
from numpy import arctan2, exp, sqrt, sin, cos, empty, arctan, tanh, tan, pi
from numpy import heaviside as heav
import numexpr as ne


def rotation(wfn, Nx, Ny, alpha, beta, gamma):
    wfn_new = empty((3, Nx, Ny), dtype='complex128')
    U = empty((3, 3), dtype='complex128')

    # Spin-1 rotation matrix
    U[0, 0] = exp(-1j * (alpha + gamma)) * (cos(beta / 2)) ** 2
    U[0, 1] = -exp(-1j * alpha) / sqrt(2.) * sin(beta)
    U[0, 2] = exp(-1j * (alpha - gamma)) * (sin(beta / 2)) ** 2
    U[1, 0] = exp(-1j * gamma) / sqrt(2.) * sin(beta)
    U[1, 1] = cos(beta)
    U[1, 2] = -exp(1j * gamma) / sqrt(2) * sin(beta)
    U[2, 0] = exp(1j * (alpha - gamma)) * (sin(beta / 2)) ** 2
    U[2, 1] = exp(1j * alpha) / sqrt(2.) * sin(beta)
    U[2, 2] = exp(1j * (alpha + gamma)) * (cos(beta / 2)) ** 2

    wfn_new[0, :, :] = U[0, 0] * wfn[0, :, :] + U[0, 1] * wfn[1, :, :] + U[0, 2] * wfn[2, :, :]
    wfn_new[1, :, :] = U[1, 0] * wfn[0, :, :] + U[1, 1] * wfn[1, :, :] + U[1, 2] * wfn[2, :, :]
    wfn_new[2, :, :] = U[2, 0] * wfn[0, :, :] + U[2, 1] * wfn[1, :, :] + U[2, 2] * wfn[2, :, :]
    return wfn_new[0, :, :], wfn_new[1, :, :], wfn_new[2, :, :]


def get_phase(N_vort, Nx, Ny, X, Y, len_x, len_y):

    def get_positions(array, number_of_vortices):
        for num in range(number_of_vortices):
            random_number = np.random.uniform(-len_x/2, len_x/2)
            for element in array:
                while abs(element - random_number) < 2:
                    random_number = np.random.uniform(-len_x/2, len_x/2)
            array.append(random_number)

    # Phase initialisation
    pos = []
    get_positions(pos, N_vort * 2)  # Array of vortex positions
    theta_k = np.empty((N_vort, Nx, Ny))
    theta_tot = np.empty((Nx, Ny))

    for k in range(N_vort // 2):
        y_m, y_p = pos[k], pos[N_vort + k]  # y-positions
        x_m, x_p = pos[N_vort // 2 + k], pos[3 * N_vort // 2 + k]  # x-positions

        # Scaling positional arguments
        Y_minus = ne.evaluate("2 * pi * (Y - y_m) / len_y")
        X_minus = ne.evaluate("2 * pi * (X - x_m) / len_x")
        Y_plus = ne.evaluate("2 * pi * (Y - y_p) / len_y")
        X_plus = ne.evaluate("2 * pi * (X - x_p) / len_x")
        x_plus = ne.evaluate("2 * pi * x_p / len_x")
        x_minus = ne.evaluate("2 * pi * x_m / len_x")

        heav_xp = heav(X_plus, 1.)
        heav_xm = heav(X_minus, 1.)

        for nn in np.arange(-5, 5):
            theta_k[k, :, :] += ne.evaluate("arctan(tanh((Y_minus + 2 * pi * nn) / 2) * tan((X_minus - pi) / 2)) "
                                            "- arctan(tanh((Y_plus + 2 * pi * nn) / 2) * tan((X_plus - pi) / 2)) "
                                            "+ pi * (heav_xp - heav_xm)")

        theta_k[k, :, :] -= ne.evaluate("(2 * pi * Y / len_y) * (x_plus - x_minus) / (2 * pi)")
        theta_tot += theta_k[k, :, :]
    return theta_tot


def polarCoreless(Nx, Ny, X, Y, c0, c1, psi_plus, psi_0, psi_minus, alpha, beta):

    rho2 = X ** 2 + Y ** 2

    if c1 <= 0.:
        Rtf = (4. * (c0 + c1) / pi) ** 0.25
    else:
        Rtf = (4. * c0 / pi) ** 0.25

    phi = arctan2(Y + 0.75, X + 0.75)
    zeta_plus = exp(1j * phi) / sqrt(2.) * (-exp(-1j * alpha) * sin(beta))
    zeta_0 = exp(1j * phi) * cos(beta)
    zeta_minus = exp(1j * phi) / sqrt(2.) * (exp(1j * alpha) * sin(beta))

    for i in range(Nx):
        for j in range(Ny):
            if Rtf ** 2 - rho2[i, j] >= 0:
                TF = (2 / pi * (Rtf ** 2 - rho2[i, j])) / Rtf ** 4

                psi_plus[i, j] = sqrt(TF) * zeta_plus[i, j]
                psi_0[i, j] = sqrt(TF) * zeta_0[i, j]
                psi_minus[i, j] = sqrt(TF) * zeta_minus[i, j]

            else:
                psi_plus[i, j] = 0 + 0j
                psi_0[i, j] = 0 + 0j
                psi_minus[i, j] = 0 + 0j
