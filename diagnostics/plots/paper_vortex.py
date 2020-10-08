import h5py
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 14})


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


vortex_data = h5py.File('../../data/paper_vortex_data.hdf5', 'r')
sqv_data = vortex_data['sqv_number'][...]
hqv_data = vortex_data['hqv_number'][...]
time_array = np.array([10000 + i * 100 for i in range(len(sqv_data))])

sqv_data = running_mean(sqv_data, 101)

fig, ax = plt.subplots(1, figsize=(6, 6))
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$N_\mathrm{vort}$')
ax.loglog(time_array[50:-50], sqv_data, 'kD', label='SQV', markersize=3)

hqv_data = running_mean(hqv_data, 101)
ax.loglog(time_array[50:-50], hqv_data, 'rD', label='HQV', markersize=3)
ax.loglog(time_array[50:-50], 8.5e3 * (time_array[50:-50] / np.log(time_array[50:-50]/100)) ** (-1/2), 'k--'
          , label=r'$[t / \ln (t/t_0)]^{-1/2}$')
ax.legend()
plt.tight_layout()
# ax.set_xticks([10000, 50000, 100000])
# ax.set_xticklabels(['10000', '50000', '100000'])
plt.savefig('../../images/unsorted/vortex_decay.png')
plt.show()
