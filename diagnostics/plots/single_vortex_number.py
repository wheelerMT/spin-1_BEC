import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({'font.size': 16})
matplotlib.use('TkAgg')

# Load in vortex count data:
filename = 'scalar_imp_gas'
vortex_count = h5py.File('../data/vortexData/{}_VD.hdf5'.format(filename), 'r')
vortex_number = vortex_count['vortex_number'][...]

time_array = np.array([200 + i * 200 for i in range(len(vortex_number))])

# Set up plots:
fig, ax = plt.subplots(1, 2, figsize=(20, 15))
ax[0].set_ylabel(r'$N_{vort}$')
ax[0].set_xlabel(r'$t/\tau$')
ax[1].set_ylabel(r'$\ell_d$')
ax[1].set_xlabel(r'$t/\tau$')
# ax.set_ylim(bottom=5e1, top=5e2)

ax[0].loglog(time_array[:], vortex_number[:], marker='D', linestyle='--', color='r')
ax[0].loglog(time_array[10:], 0.5e5 * (time_array[10:] ** (-1./2)), 'k-', label=r'$t^{-\frac{1}{2}}$')
ax[0].loglog(time_array[600:], 2.8e7 * (time_array[600:] ** (-1.)), 'k--', label=r'$t^{-1}$')
ax[0].loglog(time_array[30:300], 2e4 * (time_array[30:300] ** (-2./5)), 'k:', label=r'$t^{-\frac{2}{5}}$')
ax[0].legend()

ax[1].loglog(time_array[:], 1 / np.sqrt(vortex_number[:]), marker='D', linestyle='--', color='r')
ax[1].loglog(time_array[10:], 0.4e-2 * (time_array[10:] ** (1./4)), 'k-', label=r'$t^{\frac{1}{4}}$')
ax[1].loglog(time_array[1000:], 2.0e-4 * (time_array[1000:] ** (1./2)), 'k--', label=r'$t^{\frac{1}{2}}$')
ax[1].loglog(time_array[10:100], 0.6e-2 * (time_array[10:100] ** (1./5)), 'k:', label=r'$t^{\frac{1}{5}}$')
ax[1].legend()
plt.tight_layout()

plt.savefig('../../plots/scalar/{}_Nvort.png'.format(filename))
plt.show()
