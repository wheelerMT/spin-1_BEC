import h5py
import numpy as np
from matplotlib import pyplot as plt, animation


def animate(i):
    i *= 2  # Do every 2 frames
    print('Animating frame number {}'.format(i))
    psi_plus_plot.set_array(abs(psi_plus[:-1, :-1, i].flatten()) ** 2)
    psi_0_plot.set_array(abs(psi_0[:-1, :-1, i].flatten()) ** 2)
    psi_minus_plot.set_array(abs(psi_minus[:-1, :-1, i].flatten()) ** 2)


# Load in data
print('Loading data...')
filename = input('Enter filename: ')
data_file = h5py.File('../../../scratch/data/spin-1/kibble-zurek/{}.hdf5'.format(filename), 'r')
x, y = data_file['grid/x'], data_file['grid/y']
X, Y = np.meshgrid(x, y, indexing='ij')

dt = data_file['time/dt']
N_steps = data_file['time/Nt'][...] // data_file['time/Nframe']

psi_plus = data_file['wavefunction/psi_plus']
psi_0 = data_file['wavefunction/psi_0']
psi_minus = data_file['wavefunction/psi_minus']

num_of_frames = psi_plus.shape[-1] // 2

# Set up figure
fig, ax = plt.subplots(1, 3, figsize=(5 * 3, 4), sharey=True)
for axis in ax:
    axis.set_aspect('equal')
    axis.set_xlabel(r'$x/\xi_s$')
ax[0].set_ylabel(r'$y/\xi_s$')
ax[0].set_title(r'$|\psi_+|^2$')
ax[1].set_title(r'$|\psi_0|^2$')
ax[2].set_title(r'$|\psi_-|^2$')

# Do initial plot
print('Performing initial plot...')
psi_plus_plot = ax[0].pcolorfast(X, Y, abs(psi_plus[:-1, :-1, 0]) ** 2, vmin=0, vmax=1, cmap='jet')
psi_0_plot = ax[1].pcolorfast(X, Y, abs(psi_0[:-1, :-1, 0]) ** 2, vmin=0, vmax=1, cmap='jet')
psi_minus_plot = ax[2].pcolorfast(X, Y, abs(psi_minus[:-1, :-1, 0]) ** 2, vmin=0, vmax=1, cmap='jet')
fig.colorbar(psi_minus_plot, fraction=0.042)

print('Creating animation')
anim = animation.FuncAnimation(fig, animate, frames=num_of_frames)
anim.save('../../videos/{}.mp4'.format(filename), writer=animation.FFMpegWriter(fps=30))
