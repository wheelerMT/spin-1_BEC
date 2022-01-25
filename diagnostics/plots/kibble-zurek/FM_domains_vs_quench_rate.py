import h5py
import numpy as np
import matplotlib.pyplot as plt

quench_rates = [i for i in range(500, 10500, 500)]
filename_prefix = '1d_swislocki'
domain_count = []

for quench in quench_rates:
    domain_count_1 = 0
    domain_count_2 = 0

    with h5py.File(f'../../../scratch/data/spin-1/kibble-zurek/{filename_prefix}_{quench}.hdf5', 'r') as data_file:
        if "swislocki" in filename_prefix:
            psi_1 = data_file['wavefunction/psi_plus'][:, -1]
            psi_2 = data_file['wavefunction/psi_0'][:, -1]
        else:
            psi_1 = data_file['wavefunction/psi_plus'][:, -1]
            psi_2 = data_file['wavefunction/psi_minus'][:, -1]

        peaks_1 = np.where(abs(psi_1) ** 2 > 0.75)
        peaks_2 = np.where(abs(psi_2) ** 2 > 0.75)

        for i in range(np.size(peaks_1) - 1):
            if abs(peaks_1[0][i] - peaks_1[0][i + 1]) > 2:
                domain_count_1 += 1
        for i in range(np.size(peaks_2) - 1):
            if abs(peaks_2[0][i] - peaks_2[0][i + 1]) > 2:
                domain_count_2 += 1

    domain_count.append(domain_count_1 + domain_count_2)

fig, ax = plt.subplots(1, )
ax.loglog(quench_rates, domain_count, 'ko')
if "swislocki" in filename_prefix:
    ax.loglog(quench_rates, 6e3 * np.array(quench_rates) ** (-2 / 3), 'k--', label=r'$\tau_Q^{-2/3}$')
else:
    ax.loglog(quench_rates, 0.9e3 * np.array(quench_rates) ** (-1 / 3), 'k--', alpha=0.2, label=r'$\tau_Q^{-1/3}$')
    ax.loglog(quench_rates, 4.5e2 * np.array(quench_rates) ** (-1 / 4), 'k:', label=r'$\tau_Q^{-1/4}$')
ax.set_ylabel(r'$N_d$')
ax.set_xlabel(r'$\tau_Q$')
ax.legend()
plt.savefig(f'../../images/{filename_prefix}_domains.png', bbox_inches='tight')
