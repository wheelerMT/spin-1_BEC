import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Load in diagnostics data
diag_file = h5py.File('../../../data/1d_kibble-zurek/diagnostics/1d_polar-BA-FM_diag.hdf5', 'r')
quench_rates = diag_file['quench_rates'][...]
domains_var = []
spins_var = []
for quench in quench_rates:
    domains = diag_file['FM_domains/{}'.format(quench)][...]
    spin_windings = diag_file['spin_winding/{}'.format(quench)][...]

    domains_var.append(np.var(domains))
    spins_var.append(np.var(spin_windings))

fig, ax = plt.subplots(2, )
ax[0].loglog(quench_rates, spins_var, 'ko')
ax[0].set_xlabel(r'$\tau_Q$')
ax[0].set_ylabel(r'$<w^2>$')

ax[1].loglog(quench_rates, domains_var, 'ko')
ax[1].set_xlabel(r'$\tau_Q$')
ax[1].set_ylabel(r'$<N_d^2>$')

plt.tight_layout()
plt.show()
