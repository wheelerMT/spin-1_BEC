import matplotlib
import matplotlib.pyplot as plt
import h5py
matplotlib.use('TkAgg')
# plt.rcParams.update({'font.size': 18})


# Open required data:
filename = 'frames/100kf_1024nomag_newinit'
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
Mx = len(wvn)

# ---------------------------------------------------------------------------------------------------------------------
# Plotting occupation numbers
# ---------------------------------------------------------------------------------------------------------------------
frame = -1

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(3.5, 6))

for ax in axes:
    plt.setp(ax.spines.values(), linewidth=0.5)
    if ax == axes[-1]:
        ax.set_xlabel(r'$k\xi$')
    ax.set_ylabel(r'$n(k)$')
    ax.set_ylim(top=4e9, bottom=1e0)

# Occupation number:
axes[0].loglog(wvn, e_occ[:Mx, frame], color='k', marker='D', markersize=1, linestyle='None', label='$n(k)$')

# Sum of energies:
axes[0].loglog(wvn, sum_of[:Mx, frame], color='g', marker='o', markersize=1, linestyle='None', label=r'$n_{sum}(k)$')

# Quantum pressure:
axes[0].loglog(wvn, e_q[:Mx, frame], color='m', marker='*', markersize=1, linestyle='None', label=r'$n_q(k)$')

# Incompressible:
axes[0].loglog(wvn, e_vi[:Mx, frame], color='r', marker='^', markersize=1, linestyle='None', label=r'$n_i(k)$')

# Compressible:
axes[0].loglog(wvn, e_vc[:Mx, frame], color='b', marker='v', markersize=1, linestyle='None', label=r'$n_c(k)$')

# Spin:
axes[0].loglog(wvn, e_s[:Mx, frame], color='c', marker='P', markersize=1, linestyle='None', label=r'$n_s(k)$')

# Nematic:
axes[0].loglog(wvn, e_n[:Mx, frame], color='y', marker='s', markersize=1, linestyle='None', label=r'$n_n(k)$')

# axes[0].set_title('Uncontrolled - HQVs')

# k-lines:
axes[0].loglog(wvn[40:], 3e4 * wvn[40:] ** (-2), 'k', label=r'$k^{-2}$', linewidth=1)
axes[0].loglog(wvn[1:30], 2e3 * wvn[1:30] ** (-4), 'k--', label=r'$k^{-4}$', linewidth=1)

axes[0].legend(loc=3, fontsize='xx-small')

# Open required data:
filename = 'frames/100kf_1024nomag_psi_0=0'
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

# Occupation number:
axes[1].loglog(wvn, e_occ[:Mx, frame], color='k', marker='D', markersize=1, linestyle='None', label='$n(k)$')

# Sum of energies:
axes[1].loglog(wvn, sum_of[:Mx, frame], color='g', marker='o', markersize=1, linestyle='None', label=r'$n_{sum}(k)$')

# Quantum pressure:
axes[1].loglog(wvn, e_q[:Mx, frame], color='m', marker='*', markersize=1, linestyle='None', label=r'$n_q(k)$')

# Incompressible:
axes[1].loglog(wvn, e_vi[:Mx, frame], color='r', marker='^', markersize=1, linestyle='None', label=r'$n_i(k)$')

# Compressible:
axes[1].loglog(wvn, e_vc[:Mx, frame], color='b', marker='v', markersize=1, linestyle='None', label=r'$n_c(k)$')

# Spin:
axes[1].loglog(wvn, e_s[:Mx, frame], color='c', marker='P', markersize=1, linestyle='None', label=r'$n_s(k)$')

# Nematic:
axes[1].loglog(wvn, e_n[:Mx, frame], color='y', marker='s', markersize=1, linestyle='None', label=r'$n_n(k)$')

# axes[1].set_title('Uncontrolled - SQVs')

# k-lines:
axes[1].loglog(wvn[40:], 3.5e4 * wvn[40:] ** (-2), 'k', label=r'$k^{-2}$', linewidth=1)
# axes.loglog(wvn[1:30], 1e2 * wvn[1:30] ** (-3), 'k--', label=r'$k^{-3}$')
axes[1].loglog(wvn[1:30], 3e3 * wvn[1:30] ** (-4), 'k--', label=r'$k^{-4}$', linewidth=1)

plt.tight_layout()

save_fig = 'y'  # input('Do you wish to save the figure? (y/n): ')
if save_fig == 'y':
    fig_name = 'allsims_spectra'  #  input('Enter file name: ')
    plt.savefig('../../images/unsorted/{}.pdf'.format(fig_name))
plt.show()
