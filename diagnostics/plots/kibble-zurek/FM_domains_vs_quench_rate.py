import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Load in diagnostics data
diag_file = h5py.File('../../../data/1d_kibble-zurek/diagnostics/1d_polar-BA-FM_diag.hdf5', 'r')
quench_rates = diag_file['quench_rates'][...]
fm_domains = diag_file['FM_domains/average'][...]

# Calculate standard deviation for each quench:
fm_domains_sd = []
for quench in quench_rates:
    domains_ens = diag_file['FM_domains/{}'.format(quench)]
    fm_domains_sd.append(np.var(domains_ens))

fig, ax = plt.subplots(1, )
ax.set_xscale("log")
ax.set_yscale("log")
ax.errorbar(quench_rates, fm_domains, yerr=fm_domains_sd, capsize=5, ecolor='k', fmt='none')
ax.loglog(quench_rates, fm_domains, 'ko')
ax.loglog(quench_rates[1:], 1.9e3 * quench_rates[1:] ** (-1 / 3), 'k--', alpha=0.2, label=r'$\tau_Q^{-1/3}$')
ax.loglog(quench_rates[:], 1.15e3 * quench_rates[:] ** (-1 / 4), 'k:', label=r'$\tau_Q^{-1/4}$')
ax.set_xlim(0.8e2, 1e3)
ax.set_ylabel(r'$N_d$')
ax.set_xlabel(r'$\tau_Q$')
ax.legend()
plt.tight_layout()
plt.show()
