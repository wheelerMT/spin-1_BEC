import h5py

quenches = [i for i in range(200, 1000, 100)] + [i for i in range(1000, 8500, 500)]
runs = [i for i in range(1, 51)]

diag_file = h5py.File(
    '../../scratch/data/spin-1/kibble-zurek/diagnostics/1d_BA-FM_Q_a.hdf5', 'w')

for quench in quenches:
    print(f'On quench {quench}')
    Q_a = []
    for run in runs:
        with h5py.File(
                f'../../scratch/data/spin-1/kibble-zurek/ensembles/tau_q={quench}/1d_BA-FM_{run}.hdf5',
                'r') as file:
            Q_a.append(file['Q_a'][...])

    diag_file.create_dataset(f'{quench}/Q_a', data=Q_a)
