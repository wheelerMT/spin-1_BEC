import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams["text.usetex"] = True
plt.rc("text.latex", preamble=r"\usepackage{txfonts}")
plt.rcParams.update({"font.size": 16})

k = np.linspace(-2.5, 2.5, 100)
q = np.linspace(-2.5, 2.5, 100)
K, q = np.meshgrid(k, q)

eps_k = K**2 / 2
E_fz = np.sqrt(eps_k * (eps_k + q), dtype="complex")

c0 = 10
c1 = -0.5
n = 1
c_q = q**2 / (16 * n**2 * c1)

E_1 = np.sqrt(
    (n**2 * (c0 + 3 * c1) ** 2 - 4 * n**2 * c_q * (c0 + 2 * c1))
    * eps_k**2
    - 4 * n**3 * c1 * (c0 + 3 * c1) * (c1 - c_q) * eps_k
    + (2 * n**2 * c1 * (c1 - c_q)) ** 2
)
E_p = np.sqrt(
    eps_k**2 + n * (c0 - c1) * eps_k + 2 * n**2 * c1 * (c1 - c_q) + E_1,
    dtype="complex",
)
E_m = np.sqrt(
    eps_k**2 + n * (c0 - c1) * eps_k + 2 * n**2 * c1 * (c1 - c_q) - E_1,
    dtype="complex",
)

extent = q.min(), q.max(), K.min(), K.max()
fig, ax = plt.subplots(1, 2, sharey="all")
for axis in ax:
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"$Q$")
ax[0].set_ylabel(r"$k$")

plot_fz = ax[0].imshow(
    E_fz.imag.T,
    vmin=0,
    vmax=1,
    origin="upper",
    extent=extent,
    interpolation="gaussian",
)
ax[0].plot([0, 0], [-2.5, 0], "w--")
ax[0].set_title(r'$E_{k, f_z}$', pad=9)

plot_m = ax[1].imshow(
    E_m.imag.T,
    vmin=0,
    vmax=1,
    origin="upper",
    extent=extent,
    interpolation="gaussian",
)
ax[1].plot([-2, -2], [-2.5, 0], "w--")
ax[1].plot([2, 2], [-2.5, 0], "w--")
ax[1].set_title(r'$E_{k, +}$', pad=9)

divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(plot_m, cax=cax, orientation="vertical")
cbar.set_ticks([0, 1])
cbar.set_label(r'$E(k)$', labelpad=-5)
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
cax.axis("off")

plt.subplots_adjust(wspace=0.0)
plt.savefig("../../../../plots/spin-1/bogoliubov_energies.pdf", bbox_inches="tight")
plt.show()
