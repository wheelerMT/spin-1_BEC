import numpy as np
from numpy import conj
import numexpr as ne
import h5py
from numpy.fft import fftshift
import pyfftw
import include.diag as diag


def calc_gen_nematic_vel(dens, nematic_x, nematic_y):
    return ne.evaluate("sqrt(2 * dens) / 2 * nematic_x"), ne.evaluate("sqrt(2 * dens) / 2 * nematic_y")


# ------------------------------------------------------------------------------------------------------------------
# Loading required data
# ------------------------------------------------------------------------------------------------------------------
filename = input('Enter name of data file: ')
data_file = h5py.File('../data/{}.hdf5'.format(filename), 'r')
diag_file = h5py.File('../data/diagnostics/{}_diag.hdf5'.format(filename), 'r')

# Grid data:
x, y = np.array(data_file['grid/x']), np.array(data_file['grid/y'])
Nx, Ny = x.size, y.size
dx, dy = x[1] - x[0], y[1] - y[0]
dkx = 2 * np.pi / (Nx * dx)
dky = 2 * np.pi / (Ny * dy)  # K-space spacing
kxx = np.arange(-Nx // 2, Nx // 2) * dkx
kyy = np.arange(-Nx // 2, Nx // 2) * dky
Kx, Ky = np.meshgrid(kxx, kyy)
K = ne.evaluate("sqrt(Kx ** 2 + Ky ** 2)")
wvn = kxx[Nx // 2:]

# Three component wavefunction
psi_plus = data_file['wavefunction/psi_plus']
psi_0 = data_file['wavefunction/psi_0']
psi_minus = data_file['wavefunction/psi_minus']

num_of_frames = psi_plus.shape[-1]

wfn_data = pyfftw.empty_aligned((Nx, Ny), dtype='complex64')
fft2 = pyfftw.builders.fft2(wfn_data)
ifft2 = pyfftw.builders.ifft2(wfn_data)

# ------------------------------------------------------------------------------------------------------------------
# Calculating density, velocity and spin vectors:
# ------------------------------------------------------------------------------------------------------------------
# Density:
print('Calculating density...')
n = np.empty((Nx, Ny, num_of_frames), dtype='float32')
for i in range(num_of_frames):
    n[:, :, i] = diag.calculate_density(psi_plus[:, :, i], psi_0[:, :, i], psi_minus[:, :, i])

# Calculating gradient of wavefunction spectrally:
print('Calculating spectral derivatives...')
grad_p_x = grad_p_y = np.empty((Nx, Ny, num_of_frames), dtype='complex64')
grad_0_x = grad_0_y = np.empty((Nx, Ny, num_of_frames), dtype='complex64')
grad_m_x = grad_m_y = np.empty((Nx, Ny, num_of_frames), dtype='complex64')

for i in range(num_of_frames):
    grad_p_x[:, :, i], grad_p_y[:, :, i] = diag.spectral_derivative(psi_plus[:, :, i], Kx, Ky, fft2, ifft2)
    grad_0_x[:, :, i], grad_0_y[:, :, i] = diag.spectral_derivative(psi_0[:, :, i], Kx, Ky, fft2, ifft2)
    grad_m_x[:, :, i], grad_m_y[:, :, i] = diag.spectral_derivative(psi_minus[:, :, i], Kx, Ky, fft2, ifft2)

# Mass current:
print('Calculating mass current...')
v_mass_x, v_mass_y = np.empty((Nx, Ny, num_of_frames), dtype='float32'), \
                     np.empty((Nx, Ny, num_of_frames), dtype='float32')

for i in range(num_of_frames):
    v_mass_x[:, :, i], v_mass_y[:, :, i] = \
        diag.calculate_mass_current(psi_plus[:, :, i], psi_0[:, :, i], psi_minus[:, :, i],
                                    grad_p_x[:, :, i], grad_p_y[:, :, i], grad_0_x[:, :, i], grad_0_y[:, :, i],
                                    grad_m_x[:, :, i], grad_m_y[:, :, i])
    v_mass_x[:, :, i] /= n[:, :, i]
    v_mass_y[:, :, i] /= n[:, :, i]

# ------------------------------------------------------------------------------------------------------------------
# Calculating generalised velocities
# ------------------------------------------------------------------------------------------------------------------

# Total occupation:
occupation = np.empty((Nx, Ny, num_of_frames), dtype='float32')
for i in range(num_of_frames):
    occupation[:, :, i] = fftshift(fft2(psi_plus[:, :, i]) * conj(fft2(psi_plus[:, :, i]))
                                   + fft2(psi_0[:, :, i]) * conj(fft2(psi_0[:, :, i]))
                                   + fft2(psi_minus[:, :, i]) * conj(fft2(psi_minus[:, :, i]))).real / (Nx * Ny)

# -------------------------------------
# Quantum pressure, w_q
# -------------------------------------
grad_sqrtn_x = np.empty((Nx, Ny, num_of_frames), dtype='complex64')
grad_sqrtn_y = np.empty((Nx, Ny, num_of_frames), dtype='complex64')
wq_x = np.empty((Nx, Ny, num_of_frames), dtype='complex64')
wq_y = np.empty((Nx, Ny, num_of_frames), dtype='complex64')

for i in range(num_of_frames):
    # Calculating gradient of sqrt(n):
    grad_sqrtn_x[:, :, i], grad_sqrtn_y[:, :, i] = diag.spectral_derivative(np.sqrt(n[:, :, i]), Kx, Ky, fft2, ifft2)

    # Calculating quantum pressure generalised velocity, w_q:
    wq_x[:, :, i] = fftshift(fft2(grad_sqrtn_x[:, :, i])) / np.sqrt(Nx * Ny)
    wq_y[:, :, i] = fftshift(fft2(grad_sqrtn_y[:, :, i])) / np.sqrt(Nx * Ny)

# -------------------------------------
# Weighted velocity, w_i & w_c
# -------------------------------------
# Coefficients of incompressible and compressible velocities:
A_1 = ne.evaluate("1 - Kx ** 2 / K ** 2")
A_2 = ne.evaluate("1 - Ky ** 2 / K ** 2")
B = ne.evaluate("Kx * Ky / K ** 2")
C_1 = ne.evaluate("Kx ** 2 / K ** 2")
C_2 = ne.evaluate("Ky ** 2 / K ** 2")

# Calculating Fourier transform of mass velocity:
u_x = u_y = np.empty((Nx, Ny, num_of_frames), dtype='complex64')
for i in range(num_of_frames):
    u_x[:, :, i] = fftshift(fft2(np.sqrt(n[:, :, i]) * v_mass_x[:, :, i])) / np.sqrt(Nx * Ny)
    u_y[:, :, i] = fftshift(fft2(np.sqrt(n[:, :, i]) * v_mass_y[:, :, i])) / np.sqrt(Nx * Ny)
    print('On velocity: %i' % i)

# Incompressible:
ui_x = ui_y = np.empty((Nx, Ny, num_of_frames), dtype='complex64')
for i in range(num_of_frames):
    u_xx = u_x[:, :, i]
    u_yy = u_y[:, :, i]
    ui_x[:, :, i] = ne.evaluate("A_1 * u_xx - B * u_yy")
    ui_y[:, :, i] = ne.evaluate("-B * u_xx + A_2 * u_yy")
    print('On incompressible velocity: %i' % i)

# Compressible:
uc_x = uc_y = np.empty((Nx, Ny, num_of_frames), dtype='complex64')
for i in range(num_of_frames):
    u_xx = u_x[:, :, i]
    u_yy = u_y[:, :, i]
    uc_x[:, :, i] = ne.evaluate("C_1 * u_xx + B * u_yy")
    uc_y[:, :, i] = ne.evaluate("B * u_xx + C_2 * u_yy")
    print('On compressible velocity: %i' % i)

# -------------------------------------
# Spin, w_s
# -------------------------------------
Fx, Fy, Fz = diag_file['spin/Fx'], diag_file['spin/Fy'], diag_file['spin/Fz']

ws_x_x, ws_x_y = np.empty((Nx, Ny, num_of_frames), dtype='complex64'), np.empty((Nx, Ny, num_of_frames), dtype='complex64')
ws_y_x, ws_y_y = np.empty((Nx, Ny, num_of_frames), dtype='complex64'), np.empty((Nx, Ny, num_of_frames), dtype='complex64')
ws_z_x, ws_z_y = np.empty((Nx, Ny, num_of_frames), dtype='complex64'), np.empty((Nx, Ny, num_of_frames), dtype='complex64')

# Generalised spin velocity:
for i in range(num_of_frames):
    ws_x_x[:, :, i], ws_x_y[:, :, i] = diag.spectral_derivative(Fx[:, :, i] / n[:, :, i], Kx, Ky, fft2, ifft2)
    ws_x_x[:, :, i] = fftshift(fft2(np.sqrt(n[:, :, i]) / 2 * ws_x_x[:, :, i])) / np.sqrt(Nx * Ny)
    ws_x_y[:, :, i] = fftshift(fft2(np.sqrt(n[:, :, i]) / 2 * ws_x_y[:, :, i])) / np.sqrt(Nx * Ny)

    ws_y_x[:, :, i], ws_y_y[:, :, i] = diag.spectral_derivative(Fy[:, :, i] / n[:, :, i], Kx, Ky, fft2, ifft2)
    ws_y_x[:, :, i] = fftshift(fft2(np.sqrt(n[:, :, i]) / 2 * ws_y_x[:, :, i])) / np.sqrt(Nx * Ny)
    ws_y_y[:, :, i] = fftshift(fft2(np.sqrt(n[:, :, i]) / 2 * ws_y_y[:, :, i])) / np.sqrt(Nx * Ny)

    ws_z_x[:, :, i], ws_z_y[:, :, i] = diag.spectral_derivative(Fz[:, :, i] / n[:, :, i], Kx, Ky, fft2, ifft2)
    ws_z_x[:, :, i] = fftshift(fft2(np.sqrt(n[:, :, i]) / 2 * ws_z_x[:, :, i])) / np.sqrt(Nx * Ny)
    ws_z_y[:, :, i] = fftshift(fft2(np.sqrt(n[:, :, i]) / 2 * ws_z_y[:, :, i])) / np.sqrt(Nx * Ny)

# -------------------------------------
# Nematic, w_n
# -------------------------------------
n_xx = np.empty((Nx, Ny, num_of_frames), dtype='complex64')
n_xy = np.empty((Nx, Ny, num_of_frames), dtype='complex64')
n_xz = np.empty((Nx, Ny, num_of_frames), dtype='complex64')
n_yy = np.empty((Nx, Ny, num_of_frames), dtype='complex64')
n_yz = np.empty((Nx, Ny, num_of_frames), dtype='complex64')
n_zz = np.empty((Nx, Ny, num_of_frames), dtype='complex64')

# Calculating elements of nematic tensor:
for i in range(num_of_frames):
    psi_p2d = psi_plus[:, :, i]
    psi_02d = psi_0[:, :, i]
    psi_m2d = psi_minus[:, :, i]
    n_2d = n[:, :, i]

    n_xx[:, :, i] = ne.evaluate("1 / (2 * n_2d) * ((psi_m2d + psi_p2d) * (conj(psi_m2d) "
                                "+ conj(psi_p2d)) + 2 * abs(psi_02d).real**2)")
    n_xy[:, :, i] = ne.evaluate("1j / (2 * n_2d) * (conj(psi_m2d) * psi_p2d - conj(psi_p2d) * psi_m2d)")
    n_xz[:, :, i] = ne.evaluate("sqrt(2) / (4 * n_2d) * (psi_02d * (conj(psi_p2d) - conj(psi_m2d)) "
                                "+ conj(psi_02d) * (psi_p2d - psi_m2d))")
    n_yy[:, :, i] = ne.evaluate("1 / (2 * n_2d) * ((psi_p2d - psi_m2d) * (conj(psi_p2d) "
                                "- conj(psi_m2d)) + 2 * abs(psi_02d).real ** 2)")
    n_yz[:, :, i] = ne.evaluate("1j * sqrt(2) / (4 * n_2d) * (-psi_02d * (conj(psi_p2d) + conj(psi_m2d)) "
                                "+ conj(psi_02d) * (psi_p2d + psi_m2d))")
    n_zz[:, :, i] = ne.evaluate("1 / (2 * n_2d) * (abs(psi_m2d).real**2 + abs(psi_p2d).real**2)")

# Calculating generalised nematic velocity:
w_xx_x, w_xx_y = np.empty((Nx, Ny, num_of_frames), dtype='complex64'), np.empty((Nx, Ny, num_of_frames),
                                                                                dtype='complex64')
w_xy_x, w_xy_y = np.empty((Nx, Ny, num_of_frames), dtype='complex64'), np.empty((Nx, Ny, num_of_frames),
                                                                                dtype='complex64')
w_xz_x, w_xz_y = np.empty((Nx, Ny, num_of_frames), dtype='complex64'), np.empty((Nx, Ny, num_of_frames),
                                                                                dtype='complex64')
w_yy_x, w_yy_y = np.empty((Nx, Ny, num_of_frames), dtype='complex64'), np.empty((Nx, Ny, num_of_frames),
                                                                                dtype='complex64')
w_yz_x, w_yz_y = np.empty((Nx, Ny, num_of_frames), dtype='complex64'), np.empty((Nx, Ny, num_of_frames),
                                                                                dtype='complex64')
w_zz_x, w_zz_y = np.empty((Nx, Ny, num_of_frames), dtype='complex64'), np.empty((Nx, Ny, num_of_frames),
                                                                                dtype='complex64')

# Calculating gradient of nematic tensor:
for i in range(num_of_frames):
    w_xx_x[:, :, i], w_xx_y[:, :, i] = diag.spectral_derivative(n_xx[:, :, i], Kx, Ky, fft2, ifft2)

    w_xy_x[:, :, i], w_xy_y[:, :, i] = diag.spectral_derivative(n_xy[:, :, i], Kx, Ky, fft2, ifft2)

    w_xz_x[:, :, i], w_xz_y[:, :, i] = diag.spectral_derivative(n_xz[:, :, i], Kx, Ky, fft2, ifft2)

    w_yy_x[:, :, i], w_yy_y[:, :, i] = diag.spectral_derivative(n_yy[:, :, i], Kx, Ky, fft2, ifft2)

    w_yz_x[:, :, i], w_yz_y[:, :, i] = diag.spectral_derivative(n_yz[:, :, i], Kx, Ky, fft2, ifft2)

    w_zz_x[:, :, i], w_zz_y[:, :, i] = diag.spectral_derivative(n_zz[:, :, i], Kx, Ky, fft2, ifft2)

    print('Gradient of nematic %i complete' % i)

w_xx_x_vel, w_xx_y_vel = np.empty((Nx, Ny, num_of_frames), dtype='complex64'), np.empty((Nx, Ny, num_of_frames),
                                                                                        dtype='complex64')
w_xy_x_vel, w_xy_y_vel = np.empty((Nx, Ny, num_of_frames), dtype='complex64'), np.empty((Nx, Ny, num_of_frames),
                                                                                        dtype='complex64')
w_xz_x_vel, w_xz_y_vel = np.empty((Nx, Ny, num_of_frames), dtype='complex64'), np.empty((Nx, Ny, num_of_frames),
                                                                                        dtype='complex64')
w_yy_x_vel, w_yy_y_vel = np.empty((Nx, Ny, num_of_frames), dtype='complex64'), np.empty((Nx, Ny, num_of_frames),
                                                                                        dtype='complex64')
w_yz_x_vel, w_yz_y_vel = np.empty((Nx, Ny, num_of_frames), dtype='complex64'), np.empty((Nx, Ny, num_of_frames),
                                                                                        dtype='complex64')
w_zz_x_vel, w_zz_y_vel = np.empty((Nx, Ny, num_of_frames), dtype='complex64'), np.empty((Nx, Ny, num_of_frames),
                                                                                        dtype='complex64')
for i in range(num_of_frames):
    # Calculating generalised nematic velocity:
    w_xx_x_vel[:, :, i], w_xx_y_vel[:, :, i] = calc_gen_nematic_vel(n[:, :, i], w_xx_x[:, :, i], w_xx_y[:, :, i])

    w_xy_x_vel[:, :, i], w_xy_y_vel[:, :, i] = calc_gen_nematic_vel(n[:, :, i], w_xy_x[:, :, i], w_xy_y[:, :, i])

    w_xz_x_vel[:, :, i], w_xz_y_vel[:, :, i] = calc_gen_nematic_vel(n[:, :, i], w_xz_x[:, :, i], w_xz_y[:, :, i])

    w_yy_x_vel[:, :, i], w_yy_y_vel[:, :, i] = calc_gen_nematic_vel(n[:, :, i], w_yy_x[:, :, i], w_yy_y[:, :, i])

    w_yz_x_vel[:, :, i], w_yz_y_vel[:, :, i] = calc_gen_nematic_vel(n[:, :, i], w_yz_x[:, :, i], w_yz_y[:, :, i])

    w_zz_x_vel[:, :, i], w_zz_y_vel[:, :, i] = calc_gen_nematic_vel(n[:, :, i], w_zz_x[:, :, i], w_zz_y[:, :, i])

# Transforming nematic to Fourier space:
for i in range(num_of_frames):
    w_xx_x[:, :, i] = fft2(w_xx_x_vel[:, :, i]) / np.sqrt(Nx * Ny)
    w_xx_y[:, :, i] = fft2(w_xx_y_vel[:, :, i]) / np.sqrt(Nx * Ny)

    w_xy_x[:, :, i] = fft2(w_xy_x_vel[:, :, i]) / np.sqrt(Nx * Ny)
    w_xy_y[:, :, i] = fft2(w_xy_y_vel[:, :, i]) / np.sqrt(Nx * Ny)

    w_xz_x[:, :, i] = fft2(w_xz_x_vel[:, :, i]) / np.sqrt(Nx * Ny)
    w_xz_y[:, :, i] = fft2(w_xz_y_vel[:, :, i]) / np.sqrt(Nx * Ny)

    w_yy_x[:, :, i] = fft2(w_yy_x_vel[:, :, i]) / np.sqrt(Nx * Ny)
    w_yy_y[:, :, i] = fft2(w_yy_y_vel[:, :, i]) / np.sqrt(Nx * Ny)

    w_yz_x[:, :, i] = fft2(w_yz_x_vel[:, :, i]) / np.sqrt(Nx * Ny)
    w_yz_y[:, :, i] = fft2(w_yz_y_vel[:, :, i]) / np.sqrt(Nx * Ny)

    w_zz_x[:, :, i] = fft2(w_zz_x_vel[:, :, i]) / np.sqrt(Nx * Ny)
    w_zz_y[:, :, i] = fft2(w_zz_y_vel[:, :, i]) / np.sqrt(Nx * Ny)

    print('FFT of nematic %i complete' % i)

# ---------------------------------------------------------------------------------------------------------------------
# Calculating energies
# ---------------------------------------------------------------------------------------------------------------------
# Quantum pressure:
E_q = ne.evaluate("0.5 * (abs(wq_x).real ** 2 + abs(wq_y).real ** 2)")
print('Quantum pressure energy calculated.')

# Total velocity
E_v_tot = ne.evaluate("0.5 * (abs(u_x).real ** 2 + abs(u_y).real ** 2)")
print('Total velocity energy calculated.')

# Incompressible:
E_vi = ne.evaluate("0.5 * (abs(ui_x).real ** 2 + abs(ui_y).real ** 2)")
print('Incompressible energy calculated.')

# Compressible:
E_vc = ne.evaluate("0.5 * (abs(uc_x).real ** 2 + abs(uc_y).real ** 2)")
print('Compressible energy calculated.')

# Spin:
E_s = ne.evaluate("0.5 * (abs(ws_x_x).real ** 2 + abs(ws_x_y).real ** 2 + abs(ws_y_x).real ** 2 "
                  "+ abs(ws_y_y).real ** 2 + abs(ws_z_x).real ** 2 + abs(ws_z_y).real ** 2)")

# Nematic:
E_n = ne.evaluate("0.5 * (abs(w_xx_x).real**2 + abs(w_xx_y).real**2 + abs(w_yy_x).real**2 + abs(w_yy_y).real**2 "
                  "+ abs(w_zz_x).real**2 + abs(w_zz_y).real**2 + 2 * (abs(w_xy_x).real**2 + abs(w_xy_y).real**2 "
                  "+ abs(w_xz_x).real**2 + abs(w_xz_y).real**2 + abs(w_yz_x).real**2 + abs(w_yz_y).real**2))")
print('Nematic energy calculated.')

# ---------------------------------------------------------------------------------------------------------------------
# Calculating energy spectrum
# ---------------------------------------------------------------------------------------------------------------------
box_radius = int(np.ceil(np.sqrt(Nx ** 2 + Ny ** 2) / 2) + 1)

centerx = Nx // 2
centery = Ny // 2

eps = 1e-50  # Voids log(0)

# Defining zero arrays for spectra:
e_occ = np.zeros((box_radius, num_of_frames)) + eps
e_q = np.zeros((box_radius, np.ma.size(E_q, -1))) + eps
e_vi = np.zeros((box_radius, np.ma.size(E_vi, -1))) + eps
e_vc = np.zeros((box_radius, np.ma.size(E_vc, -1))) + eps
e_s = np.zeros((box_radius, np.ma.size(E_s, -1))) + eps
e_n = np.zeros((box_radius, np.ma.size(E_n, -1))) + eps

nc = np.zeros((box_radius, np.ma.size(E_q, -1)))  # Counts the number of times we sum over a given shell

# Summing over spherical shells:
for index in range(num_of_frames):
    for kx in range(Nx):
        for ky in range(Ny):
            k = int(np.ceil(np.sqrt((kx - centerx) ** 2 + (ky - centery) ** 2)))
            nc[k, index] += 1

            e_occ[k, index] += occupation[kx, ky, index]
            e_q[k, index] += 2 * K[kx, ky] ** (-2) * E_q[kx, ky, index]
            e_vi[k, index] += 2 * K[kx, ky] ** (-2) * E_vi[kx, ky, index]
            e_vc[k, index] += 2 * K[kx, ky] ** (-2) * E_vc[kx, ky, index]
            e_s[k, index] += 2 * K[kx, ky] ** (-2) * E_s[kx, ky, index]
            e_n[k, index] += 2 * K[kx, ky] ** (-2) * E_n[kx, ky, index]

    print('On spectra: %i' % (index + 1))

for i in range(num_of_frames):
    e_occ[:, i] /= (nc[:, i] * dkx)
    e_q[:, i] /= (nc[:, i] * dkx)
    e_vi[:, i] /= (nc[:, i] * dkx)
    e_vc[:, i] /= (nc[:, i] * dkx)
    e_s[:, i] /= (nc[:, i] * dkx)
    e_n[:, i] /= (nc[:, i] * dkx)

# Save spectra data to a file:
spectra_file = h5py.File('../data/spectra/{}_spectra.hdf5'.format(filename), 'w')

spectra_file.create_dataset('wvn', data=wvn)
spectra_file.create_dataset('e_occ', data=e_occ)
spectra_file.create_dataset('e_q', data=e_q)
spectra_file.create_dataset('e_vi', data=e_vi)
spectra_file.create_dataset('e_vc', data=e_vc)
spectra_file.create_dataset('e_s', data=e_s)
spectra_file.create_dataset('e_n', data=e_n)
spectra_file.close()
