import h5py

run_numbers = [i for i in range(1, 11)]
quenches = [1000, 2500, 5000, 7500]

filename = '1d_polar-BA-FM'

for quench in quenches:
    print(f'Saving data for quench = {quench}')
    for run in run_numbers:
        with h5py.File(f'spin-1/kibble-zurek/{filename}.hdf5', 'a') as master_file:
            with h5py.File(f'spin-1/kibble-zurek/ensembles/tau_q={quench}/{filename}_{run}.hdf5', 'r') as run_file:
                # Attempt to save data shared by all runs
                try:
                    master_file.create_dataset('grid/x', data=run_file['grid/x'][...])
                    master_file.create_dataset('time/dt', data=run_file['time/dt'][...])
                    master_file.create_dataset('time/Nframe', data=run_file['time/Nframe'][...])
                except (OSError, ValueError, RuntimeError):
                    pass

                # Save wavefunction data
                master_file.create_dataset(f'{quench}/run{run}/wavefunction/psi_plus',
                                           data=run_file['wavefunction/psi_plus'])
                master_file.create_dataset(f'{quench}/run{run}/wavefunction/psi_0',
                                           data=run_file['wavefunction/psi_0'])
                master_file.create_dataset(f'{quench}/run{run}/wavefunction/psi_minus',
                                           data=run_file['wavefunction/psi_minus'])
