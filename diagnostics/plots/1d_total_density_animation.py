import h5py
from matplotlib import pyplot as plt, animation


def animate(i):
    print('Animating frame number {}'.format(i))
    total_dens = abs(psi_plus[:, i]) ** 2 + abs(psi_0[:, i]) ** 2 + abs(psi_minus[:, i]) ** 2
    dens_plot.set_ydata(total_dens.flatten())
    plt.title(f't={int(time[i])}')


# Load in data
print('Loading data...')
filename = '1d_swislocki_5000'
data_file = h5py.File('../../data/1d_kibble-zurek/{}.hdf5'.format(filename), 'r')
x = data_file['grid/x'][...]

psi_plus = data_file['wavefunction/psi_plus']
psi_0 = data_file['wavefunction/psi_0']
psi_minus = data_file['wavefunction/psi_minus']

num_of_frames = psi_plus.shape[-1]
time = data_file['time/t'][:, 0]

# Set up figure
fig, ax = plt.subplots(1, figsize=(5, 4))
ax.set_ylim(0, 1.1)
ax.set_xlabel(r'$x/\xi_s$')
ax.set_ylabel(r'$n$')

# Do initial plot
print('Performing initial plot...')
total_dens_init = abs(psi_plus[:, 0]) ** 2 + abs(psi_0[:, 0]) ** 2 + abs(psi_minus[:, 0]) ** 2
print(total_dens_init)
dens_plot, = ax.plot(x, total_dens_init, 'k')
# fig.colorbar(dens_plot, fraction=0.042)

print('Creating animation')
anim = animation.FuncAnimation(fig, animate, frames=num_of_frames)
anim.save('../../../plots/spin-1/{}_total_dens.mp4'.format(filename), writer=animation.FFMpegWriter(fps=30))
