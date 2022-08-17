import h5py
import numpy as np

tau_q = 2500
filename_prefix = f'1d_BA-FM_{tau_q}'
runs = [i for i in range(1, 51)]
diag_file = h5py.File(f'../../scratch/data/spin-1/kibble-zurek/diagnostics/{filename_prefix}_fluctuation_diff.hdf5',
                      'w')

for run in runs:
    with h5py.File(f'../../scratch/data/spin-1/kibble-zurek/ensembles/tau_q={tau_q}/1d_BA-FM_{run}.hdf5',
                   'r') as data_file:
        print(f'On run {run}')
        # Load in required data
        psi_plus = np.fft.fft(data_file['wavefunction/psi_plus'][...], axis=0)
        psi_minus = np.fft.fft(data_file['wavefunction/psi_minus'][...], axis=0)
        X = data_file['grid/x']
        nx = len(X)

        wave_num = 5
        difference = ((psi_plus[wave_num, :]) - (psi_minus[wave_num, :])) / (np.sqrt(2) * nx)

        diag_file.create_dataset(f'{run}/difference', data=difference)
