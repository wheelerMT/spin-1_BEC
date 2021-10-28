import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Load in data:
filename = '1d_kibble-zurek/single_runs/1d_polar-BA-FM_500'  # input('Enter data filename: ')
data_file = h5py.File('../../../data/{}.hdf5'.format(filename), 'r')

psi_plus = data_file['wavefunction/psi_plus']
psi_0 = data_file['wavefunction/psi_0']
psi_minus = data_file['wavefunction/psi_minus']
n = abs(psi_plus[...]) ** 2 + abs(psi_0[...]) ** 2 + abs(psi_minus[...]) ** 2

# Other variables:
X = data_file['grid/x']
dx = X[1] - X[0]

num_of_frames = psi_plus.shape[-1]

# Generate time array
dt = data_file['time/dt'][...]
Nframe = data_file['time/Nframe'][...]
time = dt * Nframe * np.arange(1, num_of_frames + 1)
frame = int(500 / (Nframe * dt))

domains = []

for i in range(frame, num_of_frames):
    domain_count_plus = 0
    domain_count_minus = 0
    psi_plus = data_file['wavefunction/psi_plus'][:, i]
    psi_minus = data_file['wavefunction/psi_minus'][:, i]
    peaks_plus = np.where(abs(psi_plus) ** 2 > 0.75)
    peaks_minus = np.where(abs(psi_minus) ** 2 > 0.75)

    for i in range(np.size(peaks_plus) - 1):
        if abs(peaks_plus[0][i] - peaks_plus[0][i + 1]) > 4:
            domain_count_plus += 1
    for i in range(np.size(peaks_minus) - 1):
        if abs(peaks_minus[0][i] - peaks_minus[0][i + 1]) > 4:
            domain_count_minus += 1

    domains.append(domain_count_plus + domain_count_minus)

plt.plot(time[frame:], domains, 'ko')
plt.xlabel(r'$t/\tau$')
plt.ylabel(r'$N_d$')
plt.show()
