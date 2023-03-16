import h5py
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 18})
# Load in domain data
ba_fm_domains = h5py.File("data/diagnostics/1d_BA-FM_domains.hdf5", "r")
polar_ba_fm_domains = h5py.File(
    "data/diagnostics/1d_polar-BA-FM_domains.hdf5", "r"
)

quenches = [i for i in range(200, 1000, 100)] + [
    i for i in range(1000, 8500, 500)
]
print(polar_ba_fm_domains.keys())

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

fig, ax = plt.subplots(1)
ax.set_ylabel(r"$N_d$")
ax.set_xlabel(r"$\tau_Q$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.loglog(quenches, n_d_ens, "ko")
ax.errorbar(quenches, n_d_ens, yerr=n_d_std, capsize=5, ecolor="k", fmt="none")
ax.loglog(
    quenches[:12],
    1150 * np.array(quenches[:12]) ** (-1 / 4),
    "k--",
    label=r"$\tau_Q^{-1/4}$",
)
ax.legend()

axin = ax.inset_axes([0.09, 0.08, 0.40, 0.40])
axin.loglog(quenches, n_d_polar_ens, "ko", markersize=4)
axin.loglog(
    quenches[:12],
    710 * np.array(quenches[:12]) ** (-1 / 4),
    "k--",
    label=r"$\tau_Q^{-1/4}$",
)
axin.tick_params(labelsize=12)
axin.set_xlabel(r"$\tau_Q$", labelpad=-12, fontsize=12, x=0.7)
axin.set_ylabel(r"$N_d$", labelpad=-14, fontsize=12)
axin.set_ylim(0.91e2, 2e2)
axin.set_yticks([1e2, 1.2e2, 1.4e2, 1.6e2, 1.8e2, 2e2])
axin.set_yticklabels(["100", "", "", "", "", "200"])
plt.savefig("../plots/spin-1/BA-FM_domains.pdf", bbox_inches="tight")
plt.show()
