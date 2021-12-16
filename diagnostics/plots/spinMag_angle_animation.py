import h5py
import numpy as np
from matplotlib import pyplot as plt, animation
import include.diag as diag


def animate(i):
    print('Animating frame number {} of {}'.format(i + 1, num_of_frames))
    i *= 2  # Do every 2 frames
    dens = abs(psi_plus[:-1, :-1, i]) ** 2 + abs(psi_0[:-1, :-1, i]) ** 2 + abs(psi_minus[:-1, :-1, i]) ** 2
    a_fx, a_fy, a_fz, a_F = diag.calculate_spin(psi_plus[:-1, :-1, i], psi_0[:-1, :-1, i], psi_minus[:-1, :-1, i], dens)
    a_F_perp = a_fx + 1j * a_fy

    F_plot.set_array(abs(a_F))
    spin_angle_plot.set_array(np.angle(a_F_perp))


# Load in data
print('Loading data...')
filename = '2d_polar-BA-FM_1000'  # input('Enter filename: ')
data_file = h5py.File('../../../scratch/data/spin-1/kibble-zurek/{}.hdf5'.format(filename), 'r')
x, y = data_file['grid/x'], data_file['grid/y']
X, Y = np.meshgrid(x, y, indexing='ij')

dt = data_file['time/dt']
N_steps = data_file['time/Nt'][...] // data_file['time/Nframe']

psi_plus = data_file['wavefunction/psi_plus']
psi_0 = data_file['wavefunction/psi_0']
psi_minus = data_file['wavefunction/psi_minus']

num_of_frames = 100 # psi_plus.shape[-1] // 2

# Set up figure
fig, ax = plt.subplots(1, 2, figsize=(5 * 2, 4), sharey=True)
for axis in ax:
    axis.set_aspect('equal')
    axis.set_xlabel(r'$x/\xi_s$')
ax[0].set_ylabel(r'$y/\xi_s$')
ax[0].set_title(r'$|\vec{F}|$')
ax[1].set_title(r'Angle($F_\perp$)')

# Do initial plot
print('Performing initial plot...')
n = abs(psi_plus[:-1, :-1, 0]) ** 2 + abs(psi_0[:-1, :-1, 0]) ** 2 + abs(psi_minus[:-1, :-1, 0]) ** 2
fx, fy, fz, F = diag.calculate_spin(psi_plus[:-1, :-1, 0], psi_0[:-1, :-1, 0], psi_minus[:-1, :-1, 0], n)
F_perp = fx + 1j * fy

F_plot = ax[0].pcolorfast(X, Y, abs(F), vmin=0, vmax=1, cmap='jet')
spin_angle_plot = ax[1].pcolorfast(X, Y, np.angle(F_perp.real), vmin=-np.pi, vmax=np.pi, cmap='jet')
plt.colorbar(F_plot, ax=ax[0], fraction=0.042)
spin_angle_cbar = plt.colorbar(spin_angle_plot, ax=ax[1], fraction=0.042)
spin_angle_cbar.set_ticks([-np.pi, 0, np.pi])
spin_angle_cbar.set_ticklabels([r'$-\pi$', '0', r'$\pi$'])

print('Creating animation')
anim = animation.FuncAnimation(fig, animate, frames=num_of_frames)
anim.save('../../videos/{}_spins.mp4'.format(filename), writer=animation.FFMpegWriter(fps=30))
