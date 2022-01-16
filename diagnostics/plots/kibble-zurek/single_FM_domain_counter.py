import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Load in data:
filename = input('Enter data filename: ')
data_file = h5py.File('../../../data/{}.hdf5'.format(filename), 'r')

psi_plus = data_file['wavefunction/psi_plus']
psi_minus = data_file['wavefunction/psi_minus']

# Generate time array
num_of_frames = psi_plus.shape[-1]
dt = data_file['time/dt'][...]
Nframe = data_file['time/Nframe'][...]
time = dt * Nframe * np.arange(1, num_of_frames + 1)
quench_rate = 500
frame = int(quench_rate / (Nframe * dt))

domains_plus = []
domains_minus = []
for i in range(frame, num_of_frames):
    peaks_plus = np.where(abs(psi_plus[:, i]) ** 2 > 0.75)
    peaks_minus = np.where(abs(psi_minus[:, i]) ** 2 > 0.75)

    domain_count_plus = 0
    domain_count_minus = 0
    for j in range(np.size(peaks_plus) - 1):
        if abs(peaks_plus[0][j] - peaks_plus[0][j + 1]) > 4:
            domain_count_plus += 1
    for j in range(np.size(peaks_minus) - 1):
        if abs(peaks_minus[0][j] - peaks_minus[0][j + 1]) > 4:
            domain_count_minus += 1
    domains_plus.append(domain_count_plus)
    domains_minus.append(domain_count_minus)

plt.plot(time[frame:], domains_plus, 'ko', label='Plus domains')
plt.plot(time[frame:], domains_minus, 'ro', label='Minus domains')
plt.xlabel(r'$t/\tau$')
plt.ylabel(r'$N_d$')
plt.legend()
plt.tight_layout()
plt.show()
