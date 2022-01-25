import h5py
import numpy as np
import matplotlib.pyplot as plt

quench_rates = [i for i in range(500, 10500, 500)]
filename_prefix = '1d_polar-BA-FM'
spin_domain_count = []

alpha = 0.03  # Percentage coefficient

for quench in quench_rates:
    print(f'On quench rate {quench}')

    with h5py.File(f'../../../scratch/data/spin-1/kibble-zurek/{filename_prefix}_{quench}.hdf5', 'r') as data_file:
        if "swislocki" in filename_prefix:
            psi = data_file['wavefunction/psi_0']
        else:
            psi = data_file['wavefunction/psi_plus']

        # Calculate grid parameters
        x = data_file['grid/x']
        Nx = len(x)
        dx = x[1] - x[0]
        N = Nx * dx
        L = Nx * dx
        num_of_frames = psi.shape[-1]

        if "swislocki" in filename_prefix:
            Z_t = np.empty(num_of_frames // 2 - 100)
            for i in range(100, num_of_frames // 2):
                n_0 = abs(psi[:, i]) ** 2
                atom_num_difference = n_0 - alpha * N / L
                crossings = 0
                for j in range(len(atom_num_difference) - 1):
                    if (atom_num_difference[j] > 0 and atom_num_difference[j + 1] < 0) or (
                            atom_num_difference[j] < 0 and
                            atom_num_difference[j + 1] > 0):
                        crossings += 1

                Z_t[i - 100] = crossings
        else:
            Z_t = np.empty(num_of_frames // 2)
            for i in range(num_of_frames // 2):
                n_0 = abs(psi[:, i]) ** 2
                atom_num_difference = n_0 - alpha * N / L
                crossings = 0
                for j in range(len(atom_num_difference) - 1):
                    if (atom_num_difference[j] > 0 and atom_num_difference[j + 1] < 0) or (
                            atom_num_difference[j] < 0 and
                            atom_num_difference[j + 1] > 0):
                        crossings += 1

                Z_t[i] = crossings
        spin_domain_count.append(int(max(Z_t)))

fig, ax = plt.subplots(1, )
ax.loglog(quench_rates, spin_domain_count, 'ko')
if "swislocki" in filename_prefix:
    ax.loglog(quench_rates, 9.5e3 * np.array(quench_rates) ** (-2 / 3), 'k--', label=r'$\tau_Q^{-2/3}$')
else:
    ax.loglog(quench_rates, 0.9e3 * np.array(quench_rates) ** (-1 / 3), 'k--', alpha=0.2, label=r'$\tau_Q^{-1/3}$')
    ax.loglog(quench_rates, 5.5e2 * np.array(quench_rates) ** (-1 / 4), 'k:', label=r'$\tau_Q^{-1/4}$')
ax.set_ylabel(r'$N_s$')
ax.set_xlabel(r'$\tau_Q$')
ax.legend()
plt.savefig(f'../../images/{filename_prefix}_spin_domains.png', bbox_inches='tight')
