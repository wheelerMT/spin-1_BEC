import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# Load in vortex count data:
data = h5py.File('../../data/vortexData/scalar_vortex_data.hdf5', 'r')
vortex_num = data['vortex_number'][...]
# vortex_num = running_mean(vortex_num, 100)

time_array = np.array([10000 + i * 100 for i in range(len(vortex_num))])
fig, ax = plt.subplots(1, figsize=(6.4, 6.4))
# ax.set_ylim(bottom=9e1, top=2e2)
ax.set_ylabel(r'$N_\mathrm{vort}$')
ax.set_xlabel(r'$t/\tau$')

ax.loglog(time_array[::8], vortex_num[::8], linestyle='--', marker='o', color='r')
ax.loglog(time_array[::8], 0.9e3 * (time_array[::8] ** (-1/5)), 'k--', label=r'$t^{-\frac{1}{5}}$')
ax.legend()
plt.savefig('../../../plots/spin-1/scalar_Nvort.png', bbox_inches='tight')
plt.show()
