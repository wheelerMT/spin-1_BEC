import numpy as np
import matplotlib.pyplot as plt

k = np.linspace(-2.5, 2.5, 100)
q = np.linspace(-2.5, 2.5, 100)
K, Q = np.meshgrid(k, q)


eps_k = K**2 / 2
E_fz = np.sqrt(eps_k * (eps_k + Q), dtype="complex")

c0 = 10
c1 = -0.5
n = 1

E_1 = np.sqrt(
    ((c0 + 3 * c1) * n * eps_k + 2 * (c1 * n) ** 2 * (1 - Q**2)) ** 2
    - 4 * c1 * (c0 + 2 * c1) * (n * Q * eps_k) ** 2,
    dtype="complex",
)
E_p = np.sqrt(
    eps_k**2
    + n * (c0 - c1) * eps_k
    + 2 * (n * c1) ** 2 * (1 - Q**2)
    - E_1,
    dtype="complex",
)

extent = Q.min(), Q.max(), K.min(), K.max()
fig, ax = plt.subplots(
    1,
    2,
)
plot_fz = ax[0].imshow(
    E_fz.imag.T,
    extent=extent,
    origin="upper",
    interpolation="gaussian",
    vmin=0,
    vmax=1,
)
plot_p = ax[1].imshow(
    E_p.imag.T,
    extent=extent,
    origin="upper",
    interpolation="gaussian",
)
for axis in ax:
    axis.set_aspect("auto")
plt.colorbar(plot_p, ax=ax[1])
plt.show()
