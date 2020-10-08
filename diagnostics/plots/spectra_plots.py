import matplotlib
import matplotlib.pyplot as plt
import h5py
from tabulate import tabulate
matplotlib.use('TkAgg')

# Open required data:
filename = input('Enter name of spectra file to open: ')
data_file = h5py.File('../../data/{}.hdf5'.format(filename), 'r')
spectra_file = h5py.File('../../data/spectra/{}_spectra.hdf5'.format(filename), 'r')

# Open spectra data:
wvn = spectra_file['wvn']
e_occ = spectra_file['e_occ']
e_q = spectra_file['e_q']
e_vi = spectra_file['e_vi']
e_vc = spectra_file['e_vc']
e_s = spectra_file['e_s']
e_n = spectra_file['e_n']

sum_of = e_q[...] + e_vi[...] + e_vc[...] + e_s[...] + e_n[...]  # Sum of all the contributions of spectra

# Get info about the saved times of data and prints to screen:
saved_times = data_file['saved_times']
list_of_times = []
for i in range(e_occ.shape[-1]):
    list_of_times.append([i, saved_times[i]])
print(tabulate(list_of_times, headers=["Frame #", "Time"], tablefmt="orgtbl"))

Mx = len(wvn)

# ---------------------------------------------------------------------------------------------------------------------
# Plotting occupation numbers
# ---------------------------------------------------------------------------------------------------------------------
frame = int(input('Enter the index of the frame you wish to plot: '))    # Array index

fig, axes = plt.subplots(1, figsize=(8, 8))
axes.set_xlabel(r'$k\xi$')
axes.set_ylabel(r'$n(k)$')
axes.set_ylim(top=4e9, bottom=1e0)

# Occupation number:
axes.loglog(wvn, e_occ[:Mx, frame], color='k', marker='D', markersize=3, linestyle='None', label='$n(k)$')

# Sum of energies:
axes.loglog(wvn, sum_of[:Mx, frame], color='g', marker='o', markersize=3, linestyle='None', label=r'$n_{sum}(k)$')

# Quantum pressure:
axes.loglog(wvn, e_q[:Mx, frame], color='m', marker='*', markersize=3, linestyle='None', label=r'$n_q(k)$')

# Incompressible:
axes.loglog(wvn, e_vi[:Mx, frame], color='r', marker='^', markersize=3, linestyle='None', label=r'$n_i(k)$')

# Compressible:
axes.loglog(wvn, e_vc[:Mx, frame], color='b', marker='v', markersize=3, linestyle='None', label=r'$n_c(k)$')

# Spin:
axes.loglog(wvn, e_s[:Mx, frame], color='c', marker='P', markersize=3, linestyle='None', label=r'$n_s(k)$')

# Nematic:
axes.loglog(wvn, e_n[:Mx, frame], color='y', marker='s', markersize=3, linestyle='None', label=r'$n_n(k)$')


# k-lines:
# axes.loglog(wvn[40:], 2e4 * wvn[40:] ** (-2), 'k', label=r'$k^{-2}$')
# axes.loglog(wvn[1:20], 1e2 * wvn[1:20] ** (-3), 'k--', label=r'$k^{-3}$')
# axes.loglog(wvn[1:20], 1e3 * wvn[1:20] ** (-4), 'k:', label=r'$k^{-4}$')

axes.legend(loc=3)
plt.title(r'$\tau={}$'.format(saved_times[frame]))

save_fig = input('Do you wish to save the figure? (y/n): ')
if save_fig == 'y':
    fig_name = input('Enter file name: ')
    plt.savefig('../../images/unsorted/{}.png'.format(fig_name), dpi=200)
plt.show()
