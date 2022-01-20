import h5py
import matplotlib.pyplot as plt

# Load in data:
filename = '1d_polar-BA-FM_10000'  # input('Enter data filename: ')
data_file = h5py.File('../../../scratch/data/spin-1/kibble-zurek/{}.hdf5'.format(filename), 'r')

psi_plus = data_file['wavefunction/psi_plus']
psi_0 = data_file['wavefunction/psi_0']
psi_minus = data_file['wavefunction/psi_minus']
n_0 = 1.

# Other variables:
X = data_file['grid/x']

# Construct figure
fig, ax = plt.subplots(3, )
for axis in ax:
    axis.set_ylim(0, 1.1 * n_0)
    axis.set_ylabel(r'$n$')
ax[0].set_title(r'$|\psi_+|^2$')
ax[1].set_title(r'$|\psi_0|^2$')
ax[2].set_title(r'$|\psi_-|^2$')
ax[2].set_xlabel(r'$x/\xi_s$')

# Plot results
ax[0].plot(X, abs(psi_plus[:, -1]) ** 2, 'k')
ax[1].plot(X, abs(psi_0[:, -1]) ** 2, 'k')
ax[2].plot(X, abs(psi_minus[:, -1]) ** 2, 'k')

plt.savefig('../../images/{}_comp_dens.png'.format(filename), bbox_inches='tight')
