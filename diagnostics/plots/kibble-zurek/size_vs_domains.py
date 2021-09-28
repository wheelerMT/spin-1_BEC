import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Load in data:
grid_sizes = [128, 256, 512, 1024, 2048, 4096]
domain_count_list = []
for size in grid_sizes:
    domain_count_plus = 0
    domain_count_minus = 0

    with h5py.File('../../../data/1d_kibble-zurek/differing_domains/1d_polar-BA-FM_{}.hdf5'.format(size), 'r') as data_file:
        psi_plus = data_file['wavefunction/psi_plus'][:, -1]
        psi_minus = data_file['wavefunction/psi_minus'][:, -1]
        peaks_plus = np.where(abs(psi_plus) ** 2 > 0.75)
        peaks_minus = np.where(abs(psi_minus) ** 2 > 0.75)

        for i in range(np.size(peaks_plus) - 1):
            if abs(peaks_plus[0][i] - peaks_plus[0][i + 1]) > 4:
                domain_count_plus += 1
        for i in range(np.size(peaks_minus) - 1):
            if abs(peaks_minus[0][i] - peaks_minus[0][i + 1]) > 4:
                domain_count_minus += 1

        domain_count_list.append(domain_count_plus + domain_count_minus)

plt.loglog(grid_sizes, domain_count_list, 'ko--')
plt.title(r'$\tau_Q=500$')
plt.ylabel(r'$N_d$')
plt.xlabel(r'Grid size, $\xi_s$')
plt.show()
