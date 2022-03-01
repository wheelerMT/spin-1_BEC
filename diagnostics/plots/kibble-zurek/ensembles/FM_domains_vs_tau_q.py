import h5py
import numpy as np
import matplotlib.pyplot as plt

quenches = [1000, 2500, 5000, 7500]
run_numbers = [i for i in range(1, 11)]
filename_prefix = '1d_polar-BA-FM'

averaged_domain_count = []
domain_count_std = []

with h5py.File(f'../../../scratch/data/spin-1/kibble-zurek/{filename_prefix}.hdf5', 'r') as data_file:
    for quench in quenches:
        print(f'Working on quench = {quench}')
        domain_count = []
        for run in run_numbers:
            domain_count_1 = 0
            domain_count_2 = 0

            if "swislocki" in filename_prefix:
                psi_1 = data_file[f'{quench}/run{run}/wavefunction/psi_plus'][:, -1]
                psi_2 = data_file[f'{quench}/run{run}/wavefunction/psi_0'][:, -1]
            else:
                psi_1 = data_file[f'{quench}/run{run}/wavefunction/psi_plus'][:, -1]
                psi_2 = data_file[f'{quench}/run{run}/wavefunction/psi_minus'][:, -1]

            peaks_1 = np.where(abs(psi_1) ** 2 > 0.6)
            peaks_2 = np.where(abs(psi_2) ** 2 > 0.6)

            for i in range(np.size(peaks_1) - 1):
                if abs(peaks_1[0][i] - peaks_1[0][i + 1]) > 2:
                    domain_count_1 += 1
            for i in range(np.size(peaks_2) - 1):
                if abs(peaks_2[0][i] - peaks_2[0][i + 1]) > 2:
                    domain_count_2 += 1

            domain_count.append(domain_count_1 + domain_count_2)
        domain_count_std.append(np.std(domain_count))
        averaged_domain_count.append(sum(domain_count) / len(domain_count))

fig, ax = plt.subplots(1, )
ax.set_ylabel(r'$N_d$')
ax.set_xlabel(r'$\tau_Q$')
ax.set_xscale("log")
ax.set_yscale("log")
ax.errorbar(quenches, averaged_domain_count, yerr=domain_count_std, capsize=5, ecolor='k', fmt='none')
ax.loglog(quenches, averaged_domain_count, 'ko')
coeff = averaged_domain_count[0] / np.array(quenches[0]) ** (-1/4)
ax.loglog(quenches, coeff * np.array(quenches) ** (-1/4), 'k--', label=r'$\tau_Q^{-1/4}$')

ax.legend()
plt.savefig(f'../../images/{filename_prefix}_domains_ensemble.png', bbox_inches='tight')
