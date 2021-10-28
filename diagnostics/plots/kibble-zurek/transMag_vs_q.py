import numpy as np
import h5py
import include.diag as diag
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Load in data:
diag_file = h5py.File('../../../data/1d_kibble-zurek/diagnostics/1d_polar-BA-FM_diag.hdf5', 'r')

quench_time = 2000
dt = 5e-3
Nt = 2 * 2.5 * quench_time / dt
t = -0.5 * quench_time + np.arange(1000) * dt * Nt / 1000

trans_mag = diag_file['trans_mag/{}'.format(quench_time)][...]
Q = 2 - t / quench_time

plt.plot(Q, trans_mag, 'k')
plt.plot(Q, np.where(1 - Q ** 2 / 4 >= 0, 1 - Q ** 2 / 4, None), 'r--')
plt.xlim(0)
plt.ylim(-0.05)
plt.xlabel(r'$q(t)$')
plt.ylabel(r'$M_TL$')
plt.show()

