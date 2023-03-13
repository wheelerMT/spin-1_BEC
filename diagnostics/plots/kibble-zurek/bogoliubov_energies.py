import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
ax[0].set_ylabel(r"$\mathbf{k}$")

plot_fz = ax[0].pcolormesh(q, K, E_fz.imag, vmin=0, vmax=1, shading="gouraud")
ax[0].plot([0, 0], [-2.5, 0], "w--")

plot_m = ax[1].pcolormesh(q, K, E_m.imag, vmin=0, vmax=1, shading="gouraud")
ax[1].plot([-2, -2], [-2.5, 0], "w--")
ax[1].plot([2, 2], [-2.5, 0], "w--")

divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(plot_m, cax=cax, orientation="vertical")
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
cax.axis("off")

plt.savefig("../plots/spin-1/bogoliubov_energies.png", bbox_inches="tight")
plt.show()
