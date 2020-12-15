import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

# Load in vortex count data:
vortex_count = h5py.File('../../data/vortexData/vortex_2.hdf5', 'r')
sqv_number = vortex_count['sqv_number'][...]
hqv_number = vortex_count['hqv_number'][...]

time_array = np.array([10000 + i * 100 for i in range(len(sqv_number))])
t0 = 50
fig, ax = plt.subplots(1, figsize=(6.4, 6.4))
ax.set_ylabel(r'$N_\mathrm{vort}$')
ax.set_xlabel(r'$t/t_s$')
# ax.set_ylim(bottom=5e1, top=5e2)

ax.loglog(time_array[::8], sqv_number[::8], marker='D', linestyle='--', color='k', label=f'SQV')
ax.loglog(time_array[::8], hqv_number[::8], marker='o', linestyle='--', color='r', label=f'HQV')
ax.loglog(time_array[::8], 1.8e3 * time_array[::8] ** (-1/4), 'k--', label=r'$t^{-\frac{1}{4}}$')
ax.loglog(time_array[::8], 3e3 * (time_array[::8] * np.log(time_array[::8] / t0)) ** (-1/4), 'k:',
          label=r'$[t\ln(t/t_0)]^{-\frac{1}{4}}$')
ax.loglog(time_array[100::8], 7.1e3 * (time_array[100::8] / np.log(time_array[100::8] / t0)) ** (-1/2), 'k-',
          label=r'$[t/\ln(t/t_0)]^{-\frac{1}{2}}$')

ax.legend()
plt.tight_layout()
# plt.savefig('../../../plots/spin-1/vortex_number.eps')
plt.show()
