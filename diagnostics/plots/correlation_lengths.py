import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams.update({'font.size': 14})


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# Import correlation data:
filename = '512_zeeman_5%_corr'
corr = h5py.File('../../data/correlations/{}.hdf5'.format(filename), 'r')

g_theta = corr['g_theta']
g_phi = corr['g_phi']

time = np.arange(100, 200000, 100)

# * Need to calculate L_theta and L_phi by finding point where G_theta and G_phi fall to 1/4 of their peak
l_theta = np.empty(g_theta.shape[-1])
l_phi = np.empty(g_phi.shape[-1])

for i in range(g_theta.shape[-1]):
    l_theta[i] = np.argmax(g_theta[:512, i] < g_theta[0, i] / 4)
    l_phi[i] = np.argmax(g_phi[:512, i] < g_phi[0, i] / 4)

l_theta = running_mean(l_theta, 100)

fig, ax = plt.subplots(1, figsize=(6, 6))
ax.set_ylabel(r'$L_\theta$')
ax.set_xlabel(r'$t/t_s$')
ax.loglog(time[49:-49:16], l_theta[::16], linestyle='--', marker='D', color='k')
ax.loglog(time[49:-49:16], 3.5 * (time[49:-49:16] * np.log(time[49:-49:16] / 50)) ** (1/5.5), 'k:',
          label=r'[$t\ln(t/t_0)]^{\frac{1}{5.5}}$')
ax.loglog(time[1050:-500:8], 7.5e-11 * (time[1050:-500:8] * np.log(time[1050:-500:8] / 50)) ** 2, 'k--',
          label=r'[$t\ln(t/t_0)]^{2}$')
ax.legend()
plt.tight_layout()
plt.savefig('../../../plots/spin-1/prolonged_Ltheta.eps')
plt.show()
