import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# * We want to count domains by measuring the number of density peaks
# * After the number of domains has been stabilised

# Load in data:
quench_rates = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850]
domain_count_list = []
for quench in quench_rates:
    domain_count_plus = 0
    domain_count_minus = 0

    with h5py.File('../../data/1d_kibble-zurek/1d_polar-BA-FM_{}.hdf5'.format(quench), 'r') as data_file:
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


plt.loglog(quench_rates, domain_count_list, 'ko')
plt.loglog(quench_rates, 19e2 * np.array(quench_rates) ** (-1/3), 'k--', label=r'$\tau_q^{-1/3}$')
plt.legend()
plt.tight_layout()
plt.xlim(1e2, 1e3)
plt.ylabel(r'$N_d$')
plt.xlabel(r'$\tau_Q$')
plt.show()
