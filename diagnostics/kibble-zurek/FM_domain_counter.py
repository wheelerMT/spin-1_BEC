import h5py
import numpy as np

quenches = [i for i in range(500, 8500, 500)]
runs = [i for i in range(1, 51)]

# Load in diagnostics file
quench_type = '1d_BA-FM'
diag_file = h5py.File(f'../../scratch/data/spin-1/kibble-zurek/diagnostics/{quench_type}_domains.hdf5', 'w')

# Load in data:
for quench in quenches:
    nd_plus = []
    nd_minus = []
    for run in runs:
        print(f'On quench {quench}, run {run}')
        domain_count_plus = 0
        domain_count_minus = 0
        with h5py.File(f'../../scratch/data/spin-1/kibble-zurek/ensembles/tau_q={quench}/{quench_type}_{run}.hdf5',
                       'r') as data_file:
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

            nd_plus.append(domain_count_plus)
            nd_minus.append(domain_count_minus)

    # Save to dataset
    diag_file.create_dataset(f'{quench}/nd_plus', data=nd_plus)
    diag_file.create_dataset(f'{quench}/nd_minus', data=nd_minus)
    diag_file.create_dataset(f'{quench}/nd_total', data=[sum(x) for x in zip(nd_plus, nd_minus)])
