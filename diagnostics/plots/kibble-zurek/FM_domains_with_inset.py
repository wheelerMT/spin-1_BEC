import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
plt.rcParams["text.usetex"] = True
plt.rc("text.latex")
plt.rcParams.update({"font.size": 18})
# Load in domain data
ba_fm_domains = h5py.File("../../../data/diagnostics/1d_BA-FM_domains.hdf5", "r")
polar_ba_fm_domains = h5py.File(
    "../../../data/diagnostics/1d_polar-BA-FM_domains.hdf5", "r"
)

quenches = [i for i in range(200, 1000, 100)] + [
    i for i in range(1000, 8500, 500)
]

n_d_ens = []
n_d_polar_ens = []
n_d_std = []
n_d_polar_std = []
NUM_OF_RUNS = 50
for quench in quenches:
    n_d_ens.append(np.sum(ba_fm_domains[f"{quench}/nd_total"]) / NUM_OF_RUNS)
    n_d_std.append(np.std(ba_fm_domains[f"{quench}/nd_total"]))
    n_d_polar_ens.append(np.sum(polar_ba_fm_domains[f"{quench}/nd_total"]) / 6)
    n_d_polar_std.append(np.std(polar_ba_fm_domains[f"{quench}/nd_total"]))

fig, ax = plt.subplots(1, figsize=(6.4, 3.2))
ax.set_ylabel(r"$N_d$")
ax.set_xlabel(r"$\tau_Q$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(1.0e2, 3.4e2)

ax.loglog(quenches, n_d_ens, "ko")
ax.errorbar(quenches, n_d_ens, yerr=n_d_std, capsize=5, ecolor="k", fmt="none")
ax.loglog(
    quenches[:12],
    1150 * np.array(quenches[:12]) ** (-1 / 4),
    "k--",
    label=r"$\tau_Q^{-1/4}$",
)
ax.set_yticks([1e2, 2e2, 3e2])
ax.set_yticklabels(['1', '2', '3'])
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
ax.yaxis.set_major_formatter(formatter)
ax.yaxis.set_minor_formatter(ticker.NullFormatter())
ax.legend()

axin = ax.inset_axes([0.09, 0.135, 0.40, 0.40])
axin.set_xscale("log")
axin.set_yscale("log")
axin.set_xlabel(r"$\tau_Q$", labelpad=-20, x=0.7)
axin.set_ylabel(r"$N_d$", labelpad=-14)
axin.set_ylim(0.91e2, 2e2)

axin.loglog(quenches, n_d_polar_ens, "ko", markersize=4)
axin.loglog(
    quenches[:12],
    710 * np.array(quenches[:12]) ** (-1 / 4),
    "k--",
    label=r"$\tau_Q^{-1/4}$",
)
axin.set_yticks([1e2, 2e2])
axin.set_yticklabels(['1', '2'])
axin.yaxis.set_major_formatter(formatter)
axin.yaxis.set_minor_formatter(ticker.NullFormatter())
plt.savefig("../../../../plots/spin-1/BA-FM_domains.pdf", bbox_inches="tight")
plt.show()
