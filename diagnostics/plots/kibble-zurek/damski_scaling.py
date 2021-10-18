import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Load in diagnostics data
diag_file = h5py.File('../../../data/1d_kibble-zurek/diagnostics/1d_polar-BA-FM_diag.hdf5', 'r')
quench_times = diag_file['quench_times'][...]
t_hat = diag_file['t_hat/average']

# Calculate standard deviation for each quench:
t_hat_sd = []
for quench in quench_times:
    t_hat_ens = diag_file['t_hat/{}'.format(quench)]
    t_hat_sd.append(np.std(np.log(t_hat_ens)))


fig, ax = plt.subplots(1, )
ax.set_xscale("log")
ax.errorbar(quench_times, np.log(t_hat), yerr=t_hat_sd, capsize=5, ecolor='k', fmt='none')
ax.semilogx(quench_times, np.log(t_hat), 'ko')
plt.semilogx(quench_times, np.log(np.array(quench_times) ** (5/6)), 'k--', label=r'$\ln\tau_Q^{5/6}$')
plt.semilogx(quench_times, 3 + np.log(np.array(quench_times) ** (1/3)), 'k--', alpha=0.2, label=r'$\ln\tau_Q^{1/3}$')
plt.ylabel(r'$\ln \hat{t}$')
plt.xlabel(r'$\tau_Q$')
plt.tight_layout()
plt.legend()
plt.show()
