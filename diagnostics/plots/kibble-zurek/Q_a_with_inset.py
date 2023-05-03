import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
plt.rcParams["text.usetex"] = True
plt.rc("text.latex")
plt.rcParams.update({"font.size": 16})

diag_file = h5py.File("../../../data/diagnostics/1d_BA-FM_Q_a.hdf5", "r")
quenches = [i for i in range(200, 1000, 100)] + [
    i for i in range(1000, 8500, 500)
]

# Simulation data
data_file = h5py.File("../../../data/1d_BA-FM_5000.hdf5", "r")
psi_plus_k = np.fft.fft(data_file['wavefunction/psi_plus'][...], axis=0)
psi_minus_k = np.fft.fft(data_file['wavefunction/psi_minus'][...], axis=0)
nx = len(data_file['grid/x'][...])
t = data_file['time/t'][:, 0]
fluctuation = abs(((psi_plus_k[326, :]) - (psi_minus_k[326, :])) / (np.sqrt(2) * nx))

Q_a = []
Q_a_std = []
for quench in quenches:
    print(f"On quench {quench}")
    Q_a_ens = abs(diag_file[f"{quench}/Q_a"][...])

    Q_a.append(np.sum(Q_a_ens) / len(Q_a_ens))
    Q_a_std.append(np.std(Q_a_ens))

fig, ax = plt.subplots(1, figsize=(6.4, 3.2))
ax.set_ylabel(r"$Q_a$", labelpad=-30)
ax.set_xlabel(r"$\tau_Q$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.errorbar(quenches, Q_a, yerr=Q_a_std, capsize=5, ecolor="k", fmt="none")
ax.loglog(quenches, Q_a, "ko")
ax.loglog(
    quenches,
    11.8 * np.array(quenches) ** (-1 / 2),
    "k--",
    label=r"$\tau_Q^{-1/4}$",
)
ax.set_ylim(1e-1, 1e-0)
ax.yaxis.set_minor_formatter(ticker.NullFormatter())
ax.legend()

axin = ax.inset_axes([0.09, 0.135, 0.40, 0.35])
axin.plot(t / 5000, fluctuation, 'k')
axin.set_xlim(0, 0.2)
axin.set_xticks([0, 0.2])
axin.set_ylim(0, 0.006)
axin.set_yticks([0, 0.006])
axin.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
axin.set_xlabel(r'$t/\tau_Q$', labelpad=-15)
axin.set_ylabel(r'$|a_{\mathbf{k}, f_z}|$', labelpad=-7)
plt.savefig("../../../../plots/spin-1/BA-FM_Qa_scaling.pdf", bbox_inches="tight")
plt.show()
