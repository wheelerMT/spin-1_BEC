import h5py
import numpy as np
from include.diag import calculate_spin


def angular_derivative(array, wvn):
    return np.fft.ifft(1j * wvn * np.fft.fft(array))


# Initial diagnostics data:
runs = [i for i in range(1, 11)]
quenches = [i for i in range(100, 850, 50)]
filename = '1d_polar-BA-FM'

# Create empty diagnostics file
diag_path = '../../scratch/data/spin-1/kibble-zurek/diagnostics/'
diag_file = h5py.File('{}/{}_diag.hdf5'.format(diag_path, filename), 'w')
diag_file.create_dataset('quench_times', data=quenches)
diag_file.create_dataset('num_of_runs', data=len(quenches))
diag_file.close()

averaged_domains = []
averaged_that = []

# Open dataset and calculate the diagnostics
with h5py.File('../../scratch/data/spin-1/kibble-zurek/{}.hdf5'.format(filename), 'r') as data_file:
    # Create necessary grid and time data
    dt = data_file['time/dt'][...]
    Nframe = data_file['time/Nframe'][...]
    time = dt * Nframe * np.arange(1, 200 + 1)  # Time array
    x = data_file['grid/x']
    Nx = len(x)
    dx = x[1] - x[0]
    R = Nx * dx / (2 * np.pi)  # Radius of ring
    dkx = 2 * np.pi / (Nx * dx)
    Kx = np.fft.fftshift(np.arange(-Nx // 2, Nx // 2) * dkx)

    for quench in quenches:
        print('Starting diagnostics for quench rate = {}'.format(quench))
        frame_q0 = int(quench / (Nframe * dt))  # Frame when q=0

        domain_count = []
        spin_winding = []
        t_hat = []
        trans_mag = []
        for run in runs:
            psi_plus = data_file['{}/run{}/wavefunction/psi_plus'.format(quench, run)][:, -1]
            psi_minus = data_file['{}/run{}/wavefunction/psi_minus'.format(quench, run)][:, -1]

            # Calculate FM domains
            domain_count_plus = 0
            domain_count_minus = 0
            peaks_plus = np.where(abs(psi_plus) ** 2 > 0.75)
            peaks_minus = np.where(abs(psi_minus) ** 2 > 0.75)

            for i in range(np.size(peaks_plus) - 1):
                if abs(peaks_plus[0][i] - peaks_plus[0][i + 1]) > 4:
                    domain_count_plus += 1
            for i in range(np.size(peaks_minus) - 1):
                if abs(peaks_minus[0][i] - peaks_minus[0][i + 1]) > 4:
                    domain_count_minus += 1

            domain_count.append(domain_count_plus + domain_count_minus)

            # Calculate spin winding
            frame = int(quench / (Nframe * dt))

            psi_plus = data_file['{}/run{}/wavefunction/psi_plus'.format(quench, run)][:, :]
            psi_0 = data_file['{}/run{}/wavefunction/psi_0'.format(quench, run)][:, :]
            psi_minus = data_file['{}/run{}/wavefunction/psi_minus'.format(quench, run)][:, :]
            n = abs(psi_plus) ** 2 + abs(psi_0) ** 2 + abs(psi_minus) ** 2

            # Calculate spin vectors:
            fx, fy, fz, F = calculate_spin(psi_plus, psi_0, psi_minus, n)
            F_plus = fx[:, frame] + 1j * fy[:, frame]
            F_minus = fx[:, frame] - 1j * fy[:, frame]

            dF_plus = angular_derivative(F_plus, Kx)
            dF_minus = angular_derivative(F_minus, Kx)

            integral = (R / (2j * abs(F_plus) ** 2)) * (F_minus * dF_plus - F_plus * dF_minus)
            spin_winding.append(int(dx * sum(np.real(integral)) / (2 * np.pi * 2 * np.sqrt(Nx))))

            # Calculate trans_mag
            trans_mag = dx * np.sum((abs(fx) ** 2 + abs(fy) ** 2) / (Nx * dx), axis=0)

            # Find \hat{t}
            t_hat.append(data_file['{}/run{}/t_hat'.format(quench, run)][...])

        # Take average of ensembles
        averaged_domains.append(sum(domain_count) / len(runs))
        averaged_that.append(sum(t_hat) / len(runs))

        # Save diagnostics data
        with h5py.File('{}/{}_diag.hdf5'.format(diag_path, filename), 'r+') as diag:
            diag.create_dataset('FM_domains/{}'.format(quench), data=domain_count)
            diag.create_dataset('spin_winding/{}'.format(quench), data=spin_winding)
            diag.create_dataset('t_hat/{}'.format(quench), data=t_hat)
            diag.create_dataset('trans_mag/{}'.format(quench), data=trans_mag)

            # If on last quench, save the averaged FM domain data
            if quench == quenches[-1]:
                diag.create_dataset('FM_domains/average', data=averaged_domains)
                diag.create_dataset('t_hat/average', data=averaged_that)
