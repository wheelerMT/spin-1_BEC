import h5py
import numpy as np

"""File that calculates certain correlation functions from a given dataset and saves to a new dataset."""

# Import data:
filename = input('Enter filename: ')
data_path = '../../scratch/data/spin-1/kibble-zurek/{}.hdf5'.format(filename)
data_file = h5py.File(data_path, 'r')

# Loading grid array data:
x, y = data_file['grid/x'], data_file['grid/y']
X, Y = np.meshgrid(x[:], y[:], indexing='ij')
Nx, Ny = x[:].size, y[:].size
dx, dy = x[1] - x[0], y[1] - y[0]
dkx, dky = np.pi / (Nx / 2 * dx), np.pi / (Ny / 2 * dy)
kxx = np.arange(-Nx // 2, Nx // 2) * dkx
kyy = np.arange(-Nx // 2, Nx // 2) * dky
Kx, Ky = np.meshgrid(kxx, kyy, indexing='ij')

# Three component wavefunction
psi_plus_data = data_file['wavefunction/psi_plus']
psi_0_data = data_file['wavefunction/psi_0']
psi_minus_data = data_file['wavefunction/psi_minus']

n_0 = np.sum(abs(psi_plus_data[:, :, 0]) ** 2 + abs(psi_0_data[:, :, 0]) ** 2 + abs(psi_minus_data[:, :, 0]) ** 2) \
      / (Nx * Ny)

num_of_frames = psi_plus_data.shape[-1]

# Calculate quantities for angle integration
box_radius = int(np.ceil(np.sqrt(Nx ** 2 + Ny ** 2) / 2) + 1)
center_x = Nx // 2
center_y = Ny // 2

# Generate correlations data file:
corr_file = '{}_corr'.format(filename)
corr_data_path = '../../scratch/data/spin-1/kibble-zurek/correlations/{}.hdf5'.format(corr_file)
with h5py.File(corr_data_path, 'w') as corr_data:
    corr_data.create_dataset('g_theta', (box_radius, 1), maxshape=(box_radius, None))
    corr_data.create_dataset('g_phi', (box_radius, 1), maxshape=(box_radius, None))
k = 0

# Calculate mass and spin OPs
for i in range(0, num_of_frames, 4):
    print('On frame {} out of {}'.format(i, num_of_frames))

    # Load in 2D wfn
    psi_plus, psi_0, psi_minus = psi_plus_data[:, :, i], psi_0_data[:, :, i], psi_minus_data[:, :, i]

    # Calculate mass OP
    alpha_perp = np.fft.fft2(-2 * psi_plus * psi_minus)
    alpha_perp_int = np.fft.fftshift(1 / (n_0 * Nx * dx) ** 2 * np.fft.ifft2(alpha_perp * np.conj(alpha_perp))).real
    alpha_perp_int = np.where(alpha_perp_int < 0, 0, alpha_perp_int)

    # Calculate spin OP
    conj_psi = np.conj(psi_plus) * psi_minus
    q_xx_k = np.fft.fft2(np.real(conj_psi))
    q_xy_k = np.fft.fft2(np.imag(conj_psi))
    Q_int = np.fft.fftshift(4 / (n_0 * Nx * dx) ** 2 * np.fft.ifft2(q_xx_k * np.conj(q_xx_k) + q_xy_k * np.conj(q_xy_k))).real
    Q_int = np.where(Q_int < 0, 0, Q_int)

    # Do angular integration
    angle_theta = np.zeros(box_radius, )
    angle_phi = np.zeros(box_radius, )
    nc = np.zeros(box_radius, )

    for ii in range(Nx):
        for jj in range(Ny):
            r = int(np.ceil(np.sqrt((ii - center_x) ** 2 + (jj - center_y) ** 2)))
            nc[r] += 1
            angle_theta[r] += alpha_perp_int[ii, jj]
            angle_phi[r] += Q_int[ii, jj]

    with h5py.File(corr_data_path, 'r+') as new_data:
        g_theta = new_data['g_theta']
        g_theta.resize((box_radius, k + 1))
        g_theta[:, k] = angle_theta / nc

        g_phi = new_data['g_phi']
        g_phi.resize((box_radius, k + 1))
        g_phi[:, k] = angle_phi / nc

    k += 1