import h5py
import numpy as np

# * We want to count domains by measuring the number of density peaks
# * After the number of domains has been stabilised

# Load in data:
tau_q = 1000
domain_count_plus = 0
domain_count_minus = 0
with h5py.File('../../scratch/data/spin-1/kibble-zurek/1d_polar-BA-FM_{}.hdf5'.format(tau_q), 'r') as data_file:
    psi_plus = data_file['wavefunction/psi_plus'][:, -1]
    psi_minus = data_file['wavefunction/psi_minus'][:, -1]
    peaks_plus = np.where(abs(psi_plus) ** 2 > 0.75)
    peaks_minus = np.where(abs(psi_minus) ** 2 > 0.75)

    for i in range(np.size(peaks_plus) - 1):
        if abs(peaks_plus[0][i] - peaks_plus[0][i + 1]) > 2:
            domain_count_plus += 1
    for i in range(np.size(peaks_minus) - 1):
        if abs(peaks_minus[0][i] - peaks_minus[0][i + 1]) > 2:
            domain_count_minus += 1

print(f'For tau_q = {tau_q}:')
print(f'Plus domains: {domain_count_plus}, minus domains: {domain_count_minus}, '
      f'total domains: {domain_count_plus + domain_count_minus}')
