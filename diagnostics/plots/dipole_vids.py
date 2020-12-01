import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load in data:
filename = input('Enter data filename: ')
data_file = h5py.File('../../data/{}.hdf5'.format(filename), 'r')
diag_file = h5py.File('../../data/diagnostics/{}_diag.hdf5'.format(filename), 'r')

psi_plus = data_file['wavefunction/psi_plus']
psi_0 = data_file['wavefunction/psi_0']
psi_minus = data_file['wavefunction/psi_minus']

spin_expec_mag = diag_file['spin/spin_expectation']

# Other variables:
x, y = data_file['grid/x'], data_file['grid/y']
dx, dy = x[1] - x[0], y[1] - y[0]
X, Y = np.meshgrid(x[:], y[:])

# Loading time variables:
Nt, dt, Nframe = np.array(data_file['time/Nt']), np.array(data_file['time/dt']), np.array(data_file['time/Nframe'])
num_of_frames = psi_plus.shape[-1]

# Set up figure:
fig, ax = plt.subplots(1, 3, sharey=True, figsize=(10, 10))
ax[0].set_ylabel(r'$y / \xi_s$')
ax[0].set_title(r'$|\psi_+|^2$')
ax[1].set_title(r'$|\psi_-|^2$')
ax[2].set_title(r'$|<\vec{F}>|$')
for axis in ax:
    axis.set_xlabel(r'$x / \xi_s$')

cvals_dens = np.linspace(0, 1600, 25, endpoint=True)
cvals_spin = np.linspace(0, 1, 25, endpoint=True)

# Initial frame plot:
densPlus_plot = ax[0].contourf(X, Y, abs(psi_plus[:, :, 0]) ** 2, cvals_dens, cmap='gnuplot')
densMinus_plot = ax[1].contourf(X, Y, abs(psi_minus[:, :, 0]) ** 2, cvals_dens, cmap='gnuplot')
spin_plot = ax[2].contourf(X, Y, spin_expec_mag[:, :, 0], cvals_spin, cmap='PuRd')
cont = [densPlus_plot, densMinus_plot, spin_plot]

# Set up color bars:
dens_cbar = plt.colorbar(densMinus_plot, ax=ax[1], fraction=0.044, pad=0.03)
phase_cbar = plt.colorbar(spin_plot, ax=ax[2], ticks=[0, 1], fraction=0.044, pad=0.03)

for axis in ax:
    axis.set_aspect('equal')


def animate(i):
    """Animation function for plots."""
    global cont
    for contour in cont:
        for c in contour.collections:
            c.remove()

    ax[0].contourf(X, Y, abs(psi_plus[:, :, i]) ** 2, cvals_dens, cmap='gnuplot')
    ax[1].contourf(X, Y, abs(psi_minus[:, :, i]) ** 2, cvals_dens, cmap='gnuplot')
    ax[2].contourf(X, Y, spin_expec_mag[:, :, i], cvals_spin, cmap='PuRd')

    cont = [ax[0], ax[1], ax[2]]
    print('On density iteration %i' % (i + 1))
    plt.suptitle(r'$\tau$ = %2f' % (Nframe * dt * i), y=0.7)
    return cont


# Calls the animation function and saves the result
anim = animation.FuncAnimation(fig, animate, frames=num_of_frames, repeat=False)
anim.save('../../../plots/spin-1/{}.mp4'.format(filename[7:]), dpi=200,
          writer=animation.FFMpegWriter(fps=60, codec="libx264", extra_args=['-pix_fmt', 'yuv420p']))
print('Density video saved successfully.')
