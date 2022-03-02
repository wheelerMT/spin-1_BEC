import h5py
import numpy as np
from matplotlib import pyplot as plt
from numpy import conj
from numpy.fft import fft, ifft, ifftshift
import include.diag as diag
from scipy.signal import find_peaks, savgol_filter

# Load in data
filename_prefix = '1d_polar-BA-FM'
quenches = [1000, 2500, 5000, 7500]
run_numbers = [i for i in range(1, 11)]

averaged_num_of_peaks = []
peaks_std = []

with h5py.File(f'../../../scratch/data/spin-1/kibble-zurek/{filename_prefix}.hdf5', 'r') as data_file:
    for quench in quenches:
        print(f'Working on quench = {quench}')
        num_of_peaks = []

        # Loading grid array data:
        x = data_file['grid/x'][...]
        Nx = len(x)
        for run in run_numbers:
            # Loading wavefunction data
            psi_plus = data_file[f'{quench}/run{run}/wavefunction/psi_plus'][:, -1]
            psi_0 = data_file[f'{quench}/run{run}/wavefunction/psi_0'][:, -1]
            psi_minus = data_file[f'{quench}/run{run}/wavefunction/psi_minus'][:, -1]

            # Calculate densities
            n_plus = abs(psi_plus) ** 2
            n_0 = abs(psi_0) ** 2
            n_minus = abs(psi_minus) ** 2
            n = abs(psi_plus) ** 2 + abs(psi_0) ** 2 + abs(psi_minus) ** 2

            # Calculate spin vectors
            fx, fy, fz, _ = diag.calculate_spin(psi_plus, psi_0, psi_minus, n)
            fz /= n  # Normalise spin vector

            # Calculate initial correlation
            G_fz = (1 / Nx * ifftshift(ifft(fft(fz) * conj(fft(fz))))).real
            G_fz = savgol_filter(G_fz, 151, 2)

            peaks, _ = find_peaks(G_fz, 0)
            num_of_peaks.append(len(peaks))
        peaks_std.append(np.std(num_of_peaks))
        averaged_num_of_peaks.append(sum(num_of_peaks) / len(num_of_peaks))

fig, ax = plt.subplots(1, )
ax.set_ylabel('Number of peaks')
ax.set_xlabel(r'$\tau_q$')
ax.set_xscale("log")
ax.set_yscale("log")
ax.errorbar(quenches, averaged_num_of_peaks, yerr=peaks_std, capsize=5, ecolor='k', fmt='none')
ax.loglog(quenches, averaged_num_of_peaks, 'ko')
coeff = averaged_num_of_peaks[0] / np.array(quenches[0]) ** (-1/4)
ax.loglog(quenches, coeff * np.array(quenches) ** (-1/4), 'k--', label=r'$\tau_Q^{-1/4}$')
ax.legend()
plt.savefig(f'../../images/{filename_prefix}_correlation_peaks_ensemble.png', bbox_inches='tight')
