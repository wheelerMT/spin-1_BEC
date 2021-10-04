import h5py
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Load in diagnostics data
diag_file = h5py.File('../../../data/1d_kibble-zurek/diagnostics/1d_polar-BA-FM_diag.hdf5', 'r')
quench_rates = diag_file['quench_rates'][...]
fm_domains = diag_file['FM_domains/average'][...]

plt.loglog(quench_rates, fm_domains, 'ko')
plt.loglog(quench_rates[1:], 1.9e3 * quench_rates[1:] ** (-1/3), 'k--', alpha=0.2, label=r'$\tau_Q^{-1/3}$')
plt.loglog(quench_rates[1:], 1.15e3 * quench_rates[1:] ** (-1/4), 'k:', label=r'$\tau_Q^{-1/4}$')
plt.xlim(1e2, 1e3)
plt.ylabel(r'$N_d$')
plt.xlabel(r'$\tau_Q$')
plt.legend()
plt.tight_layout()
plt.show()
