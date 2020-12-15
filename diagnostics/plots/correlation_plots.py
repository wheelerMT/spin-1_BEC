import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams.update({'font.size': 9})


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# Import correlation data:
hqv_file = 'EPP_HQV_corr'
hqv_corr = h5py.File('../../data/correlations/{}.hdf5'.format(hqv_file), 'r')
sqv_file = 'EPP_SQV_corr'
sqv_corr = h5py.File('../../data/correlations/{}.hdf5'.format(sqv_file), 'r')

g_theta_hqv = hqv_corr['g_theta']
g_phi_hqv = hqv_corr['g_phi']
g_theta_sqv = sqv_corr['g_theta']
g_phi_sqv = sqv_corr['g_phi']
time = np.arange(100, 100000, 100)

# Plots:
colors = ['r', 'g', 'b', 'y', 'c', 'm']
markers = ['D', 'o', '*', 'v', 'X', 's']
frames = [25, 75, 150, 250, 500, 1000]
radius = 130
step = 6
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
for axis in [ax1, ax2, ax3]:
    axis.set_ylim(bottom=0, top=1.1)
    axis.set_xlim(left=0, right=120)
    axis.set_xlabel(r'$r/\xi_s$')
    if axis == ax1:
        axis.set_ylabel(r'$G_\theta(r, t)$')
        axis.set_title('HQV')
    if axis == ax2:
        axis.set_ylabel(r'$G_\theta(r, t)$')
        axis.set_title('SQV')
    if axis == ax3:
        axis.set_ylabel(r'$G_\phi(r, t)$')

for i in range(6):
    ax1.plot(np.arange(0, radius, step), g_theta_hqv[:radius:step, frames[i]-1], linestyle='-', marker=markers[i],
             markersize=4, color=colors[i], label=r'$t = {} \times 10^3t_s$'.format(frames[i] / 10))
    ax2.plot(np.arange(0, radius, step), g_theta_sqv[:radius:step, frames[i] - 1], linestyle='-', marker=markers[i],
             markersize=4, color=colors[i])
    ax3.plot(np.arange(0, radius, step), g_phi_hqv[:radius:step, frames[i] - 1], linestyle='-', marker=markers[i],
             markersize=4, color=colors[i])

# * Need to calculate L_theta and L_phi by finding point where G_theta and G_phi fall to 1/4 of their peak
l_theta_hqv = np.empty(g_theta_hqv.shape[-1])
l_theta_sqv = np.empty(g_theta_sqv.shape[-1])
l_phi_hqv = np.empty(g_phi_hqv.shape[-1])

for i in range(g_theta_hqv.shape[-1]):
    l_theta_hqv[i] = np.argmax(g_theta_hqv[:512, i] < g_theta_hqv[0, i] / 4)
    l_theta_sqv[i] = np.argmax(g_theta_sqv[:512, i] < g_theta_sqv[0, i] / 4)
    l_phi_hqv[i] = np.argmax(g_phi_hqv[:512, i] < g_phi_hqv[0, i] / 4)

ax4.set_xlabel(r'$t / \tau$')
ax4.set_ylabel(r'$L / \xi_s$')
ax4.loglog(time[10::8], l_theta_hqv[10::8], linestyle='--', marker='D', markersize=3, color='b', label=r'$L_\theta$: HQV')
ax4.loglog(time[10::8], l_phi_hqv[10::8], linestyle='--', marker='D', markersize=3, color='r', label=r'$L_\phi$: HQV')
ax4.loglog(time[10::8], l_theta_sqv[10::8], linestyle='--', marker='D', markersize=3, color='g', label=r'$L_\theta$: SQV')
#ax4.loglog(time[10::8], 0.4e1 * np.sqrt((time[10::8] / 25) / np.log(time[10::8] / 25)), 'k--',
           #label=r'$[t / \ln (t/t_0)]^{1/2}$')
#ax4.loglog(time[10::8], 0.16e1 * (time[10::8] * np.log(time[10::8] / 25)) ** (1/4), 'k:',
           #label=r'$[t\ln (t/t_0)]^{1/4}$')
ax4.loglog(time[10::8], 0.26e1 * (time[10::8]) ** (1/4), 'k-', label=r'$t^{1/4}$')
ax4.loglog(time[10::8], 0.42e1 * (time[10::8]) ** (1/5), 'k-.', label=r'$t^{1/5}$')
ax4.legend(frameon=False)
ax1.legend(frameon=False)
letters = ['(a)', '(b)', '(c)', '(d)']
for i, axis in enumerate([ax1, ax2, ax3, ax4]):
    axis.set_title(letters[i], x=0.05)

plt.tight_layout()
plt.savefig('../../../plots/spin-1/correlations.png', bbox_inches='tight')
plt.show()

