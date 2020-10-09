import h5py
import matplotlib.pyplot as plt
from matplotlib import animation

"""File that generates an animated plot of various spectra."""


def animate(i):
    """Animation function that plots spectra"""
    plt.cla()

    # Occupation number:
    axes.loglog(wvn, e_occ[:Mx, i], color='k', marker='D', markersize=3, linestyle='None', label='$n(k)$')

    # Sum of energies:
    axes.loglog(wvn, sum_of[:Mx, i], color='g', marker='o', markersize=3, linestyle='None', label=r'$n_{sum}(k)$')

    # Quantum pressure:
    axes.loglog(wvn, e_q[:Mx, i], color='m', marker='*', markersize=3, linestyle='None', label=r'$n_q(k)$')

    # Incompressible:
    axes.loglog(wvn, e_vi[:Mx, i], color='r', marker='^', markersize=3, linestyle='None', label=r'$n_i(k)$')

    # Compressible:
    axes.loglog(wvn, e_vc[:Mx, i], color='b', marker='v', markersize=3, linestyle='None', label=r'$n_c(k)$')

    # Spin:
    axes.loglog(wvn, e_s[:Mx, i], color='c', marker='P', markersize=3, linestyle='None', label=r'$n_s(k)$')

    # Nematic:
    axes.loglog(wvn, e_n[:Mx, i], color='y', marker='s', markersize=3, linestyle='None', label=r'$n_n(k)$')

    # Plotting k-lines:
    # axes.loglog(wvn[50:], 2e4 * wvn[50:] ** (-2), 'k', label=r'$k^{-2}$')
    # axes.loglog(wvn[5:40], 4e3 * wvn[5:40] ** (-4), 'k:', label=r'$k^{-4}$')

    axes.set_xlabel(r'$k \xi_s$')
    axes.set_ylabel(r'$\mathcal{n}(k)$')
    axes.set_ylim(bottom=1e0, top=4e8)
    plt.legend(loc=3)
    plt.title(r'$\tau = %2f$' % (40000 + Nframe * dt * i))
    print('On spectra plot %i' % (i + 1))
    return axes


# Load in data:
filename = input('Enter filename: ')
spectra_file = h5py.File('../../data/spectra/{}_spectra.hdf5'.format(filename), 'r')

# Open spectra data:
wvn = spectra_file['wvn']   # Wavenumbers
e_occ = spectra_file['e_occ']  # Occupation number
e_q = spectra_file['e_q']   # Quantum pressure
e_vi = spectra_file['e_vi']  # Incompressible
e_vc = spectra_file['e_vc']  # Compressible
e_s = spectra_file['e_s']   # Spin
e_n = spectra_file['e_n']   # Nematic

sum_of = e_q[...] + e_vi[...] + e_vc[...] + e_s[...] + e_n[...]  # Sum of all the contributions of spectra

# Other variables:
Mx = len(wvn)
Nframe = 5000
dt = 1e-2
num_of_frames = e_occ.shape[-1]

# Set up plot
fig, axes = plt.subplots(1, figsize=(8, 8))
axes.set_xlabel(r'$k\xi$')
axes.set_ylabel(r'$n(k)$')
axes.set_ylim(top=4e8, bottom=1e0)

anim = animation.FuncAnimation(fig, animate, frames=num_of_frames)
anim.save('../../videos/{}_spectra.mp4'.format(filename), dpi=200, writer=animation.FFMpegWriter(fps=60))
