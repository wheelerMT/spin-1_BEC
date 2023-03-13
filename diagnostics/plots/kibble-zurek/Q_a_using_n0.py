import h5py
import numpy as np
import matplotlib.pyplot as plt

diag_file = h5py.File("data/diagnostics/1d_BA-FM_n0_Q_a.hdf5", "r")

quenches = [i for i in range(200, 1000, 100)] + [
    i for i in range(1000, 8500, 500)
]

Q_a = []
Q_a_std = []
for quench in quenches:
    print(f"On quench {quench}")
    Q_a_ens = diag_file[f"{quench}/Q_a"][...]

    Q_a.append(np.sum(Q_a_ens) / len(Q_a_ens))
    Q_a_std.append(np.std(Q_a_ens))

fig, ax = plt.subplots(1)

ax.set_ylabel(r"$Q_a$")
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
ax.legend()
plt.show()
