import h5py
import matplotlib
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
# matplotlib.use('TkAgg')

# Load in spectra file:
filename = input('Enter filename: ')
data_file = h5py.File('../../data/{}.hdf5'.format(filename), 'r')
spectra_file = h5py.File('../../data/spectra/{}_spectra.hdf5'.format(filename), 'r')

# Load in spectra:
wvn = spectra_file['wvn'][...]
e_occ = spectra_file['e_occ']
e_q = spectra_file['e_q']
e_vi = spectra_file['e_vi']
e_vc = spectra_file['e_vc']

# Other data:
Mx = len(wvn)

# Get info about the saved times of data and prints to screen:
saved_times = data_file['saved_times']
list_of_times = []
for i in range(e_occ.shape[-1]):
    list_of_times.append([i, saved_times[i]])
print(tabulate(list_of_times, headers=["Frame #", "Time"], tablefmt="orgtbl"))

# ---------------------------------------------------------------------------------------------------------------------
# Plotting occupation numbers
# ---------------------------------------------------------------------------------------------------------------------
sum_of = e_q[...] + e_vi[...] + e_vc[...]  # Sum of all the contributions of spectra

frame = int(input('Enter the index of the frame you want to plot: '))  # Index corresponding to the frame we want

fig, ax = plt.subplots(1, figsize=(8, 8))
ax.set_xlabel(r'$k \xi$')
ax.set_ylabel(r'$n(k)$')
ax.set_ylim(top=3.2e9, bottom=1e0)

# Occupation number:
ax.loglog(wvn, e_occ[:Mx, frame], color='k', marker='D', markersize=3, linestyle='None', label='$n(k)$')

# Sum of energies:
ax.loglog(wvn, sum_of[:Mx, frame], color='g', marker='o', markersize=3, linestyle='None', label=r'$n_{sum}(k)$')

# Quantum pressure:
ax.loglog(wvn, e_q[:Mx, frame], color='m', marker='*', markersize=3, linestyle='None', label=r'$n_q(k)$')

# Incompressible:
ax.loglog(wvn, e_vi[:Mx, frame], color='r', marker='^', markersize=3, linestyle='None', label=r'$n_i(k)$')

# Compressible:
ax.loglog(wvn, e_vc[:Mx, frame], color='b', marker='v', markersize=3, linestyle='None', label=r'$n_c(k)$')

# Plot various k:
# ax.loglog(wvn[30:Mx], 0.04 * wvn[30:Mx] ** (-2), 'k-', label=r'$k^{-2}$')
# ax.loglog(wvn[1:20], 0.00001 * wvn[1:20] ** (-3.), 'k:', label=r'$k^{-3}$')
# ax.loglog(wvn[1:60], 0.00001 * wvn[1:60] ** (-4), 'k-.', label=r'$k^{-4}$')

ax.legend(loc=3)
plt.title(r'$\tau$ = {}'.format(saved_times[frame]))

save_image = input('Do you want to save the plot? (y/n): ')
if save_image == 'y':
    image_name = input('Enter the name of the plot: ')
    plt.savefig('../../images/unsorted/{}.png'.format(image_name), dpi=200)
plt.show()
