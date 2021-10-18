import h5py
import numpy as np
import include.diag as diag

# Initial diagnostics data:
runs = [1]  # [i for i in range(1, 11)]
quenches = [0.8, 1, 6, 10, 35, 58, 88]
filename = '1d_polar-BA_damski'

# Create empty diagnostics file
diag_path = '../../scratch/data/spin-1/kibble-zurek/diagnostics/'
diag_file = h5py.File('{}/{}_diag.hdf5'.format(diag_path, filename), 'w')
diag_file.create_dataset('quench_times', data=quenches)
diag_file.create_dataset('num_of_runs', data=len(quenches))
diag_file.close()

averaged_that = []

# Open dataset and calculate the diagnostics
with h5py.File('../../scratch/data/spin-1/kibble-zurek/{}.hdf5'.format(filename), 'r') as data_file:
    # Create necessary grid and time data
    dt = data_file['time/dt'][...]
    x = data_file['grid/x']
    Nx = len(x)
    dx = x[1] - x[0]

    for quench in quenches:
        Nframe = data_file['{}/time/Nframe'.format(quench)][...]
        Nt = data_file['{}/time/Nt'.format(quench)][...]
        print('Starting diagnostics for quench time = {}'.format(quench))

        t_hat = []

        for run in runs:
            psi_plus = data_file['{}/run{}/wavefunction/psi_plus'.format(quench, run)][:, :]
            psi_0 = data_file['{}/run{}/wavefunction/psi_0'.format(quench, run)][:, :]
            psi_minus = data_file['{}/run{}/wavefunction/psi_minus'.format(quench, run)][:, :]
            n = abs(psi_plus) ** 2 + abs(psi_0) ** 2 + abs(psi_minus) ** 2

            # Calculate spin vectors:
            fx, fy, fz, F = diag.calculate_spin(psi_plus, psi_0, psi_minus, n)

            # Calculate transverse magnetisation
            trans_mag = np.empty(200)
            for i in range(200):
                trans_mag[i] = dx * np.sum(abs(fx[:, i]) ** 2 + abs(fy[:, i]) ** 2) / (Nx * dx)

            # Find \hat{t}
            t_hat.append(Nframe * dt * np.where(trans_mag >= 0.01)[0][0])

        # Take average of ensembles
        averaged_that.append(sum(t_hat) / len(runs))

        # Save diagnostics data
        with h5py.File('{}/{}_diag.hdf5'.format(diag_path, filename), 'r+') as diag:
            diag.create_dataset('t_hat/{}'.format(quench), data=t_hat)

            # If on last quench, save the averaged FM domain data
            if quench == quenches[-1]:
                diag.create_dataset('t_hat/average', data=averaged_that)
