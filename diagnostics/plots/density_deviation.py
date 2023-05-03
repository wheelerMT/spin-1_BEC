import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True
plt.rc("text.latex")
plt.rcParams.update({"font.size": 18})

# Load in data
data_file_500 = h5py.File('../../data/1d_BA-FM_500.hdf5', 'r')
psi_plus_500 = data_file_500['wavefunction/psi_plus'][...]
psi_0_500 = data_file_500['wavefunction/psi_0'][...]

data_file_1000 = h5py.File('../../data/1d_BA-FM_1000.hdf5', 'r')
psi_plus_1000 = data_file_1000['wavefunction/psi_plus'][...]
psi_0_1000 = data_file_1000['wavefunction/psi_0'][...]

data_file_5000 = h5py.File('../../data/1d_BA-FM_5000.hdf5', 'r')
psi_plus_5000 = data_file_5000['wavefunction/psi_plus'][...]
psi_0_5000 = data_file_5000['wavefunction/psi_0'][...]

# Other variables:
x = data_file_500['grid/x']
nx = len(x)
dx = x[1] - x[0]
t = data_file_500['time/t'][:, 0]
Q = -t / 500
Q[Q > 2] = 2
n_0_analytical = abs(0.5 * np.sqrt(2 + Q)) ** 2
n_1_analytical = abs(np.sqrt(2) / 4 * np.sqrt(2 - Q)) ** 2
# Averages
n_0_500_avg = dx * np.sum(abs(psi_0_500) ** 2, axis=0) / (nx * dx)
n_0_1000_avg = dx * np.sum(abs(psi_0_1000) ** 2, axis=0) / (nx * dx)
n_0_5000_avg = dx * np.sum(abs(psi_0_5000) ** 2, axis=0) / (nx * dx)
n_1_500_avg = dx * np.sum(abs(psi_plus_500) ** 2, axis=0) / (nx * dx)
n_1_1000_avg = dx * np.sum(abs(psi_plus_1000) ** 2, axis=0) / (nx * dx)
n_1_5000_avg = dx * np.sum(abs(psi_plus_5000) ** 2, axis=0) / (nx * dx)

fig, ax = plt.subplots(1, figsize=(6.4, 3.2))
ax.set_xlim(-1, 1)
ax.set_ylim(0, 0.8)
ax.set_xlabel(r'$t/\tau_Q$')
ax.set_ylabel(r'$N_0/L$')
ax.plot(-Q, n_0_500_avg, 'dodgerblue', label=r'$\tau_Q=500$')
ax.plot(-Q, n_0_1000_avg, 'palevioletred', label=r'$\tau_Q=1000$')
ax.plot(-Q, n_0_5000_avg, 'gold', label=r'$\tau_Q=5000$')
ax.plot(-Q, n_0_analytical, 'k', label='Analytical')
ax.plot([0, 0], [0, 0.5], 'k--')

axin = ax.inset_axes([0.11, 0.16, 0.35, 0.45])
axin.set_xlim(-1, 1)
axin.set_ylim(0, 0.5)
axin.set_xticks([-1, 1])
axin.set_yticks([0, 0.5])
axin.set_xlabel(r'$t/\tau_Q$', labelpad=-20)
axin.set_ylabel(r'$N_1/L$', labelpad=-20)

axin.plot(-Q, n_1_500_avg, 'dodgerblue')
axin.plot(-Q, n_1_1000_avg, 'palevioletred')
axin.plot(-Q, n_1_5000_avg, 'gold')
axin.plot(-Q, n_1_analytical, 'k')
axin.plot([0, 0], [0, 0.25], 'k--')

ax.legend(loc='upper right', prop={'size': 11.5})
plt.savefig('../../../plots/spin-1/density_deviation.pdf', bbox_inches='tight')
plt.show()
