import h5py
import matplotlib.pyplot as plt
import numpy as np

# Load in data
filename = '1d_swislocki_5000'
data_file = h5py.File('../../../scratch/data/spin-1/kibble-zurek/{}.hdf5'.format(filename), 'r')

if "swislocki" in filename:
    psi = data_file['wavefunction/psi_0']
else:
    psi = data_file['wavefunction/psi_plus']

num_of_frames = psi.shape[-1]

alpha = 0.03

# Calculate grid parameters
x = data_file['grid/x']
Nx = len(x)
dx = x[1] - x[0]
N = Nx * dx
L = Nx * dx

# Create time array
Nt = data_file['time/Nt'][...]
dt = data_file['time/dt'][...]
Nframe = data_file['time/Nframe'][...]
if "swislocki" in filename:
    time = data_file['time/t']
else:
    time = dt * Nframe * np.arange(-num_of_frames // 2, num_of_frames // 2)

# Calculate Z(t)
Z_t = np.empty(num_of_frames)
for i in range(num_of_frames):
    print(f'On frame {i}')
    n_0 = abs(psi[:, i]) ** 2
    atom_num_difference = n_0 - alpha * N / L
    crossings = 0

    for j in range(len(atom_num_difference) - 1):
        if (atom_num_difference[j] > 0 and atom_num_difference[j + 1] < 0) or (atom_num_difference[j] < 0 and
                                                                               atom_num_difference[j + 1] > 0):
            crossings += 1

    Z_t[i] = crossings

N_s_index = np.argmax(Z_t)
print(f'Time for spin domain peak is {time[N_s_index]}')
# N_s = int(max(Z_t))
# N_d = int(sum(Z_t[-21:]) / 20)
#
# # Do plot
# print('Creating figure...')
# fig, ax = plt.subplots(1, )
# ax.set_xlabel(r'Time')
# ax.set_ylabel(r'$Z(t)$')
# ax.plot(time, Z_t)
#
# # Set domains text
# ax.text(time[-400], max(Z_t), r'$N_s={}$'.format(N_s))
# ax.text(time[-400], max(Z_t) - 10, r'$N_d={}$'.format(N_d))

# plt.savefig('../../images/{}_SD.png'.format(filename), bbox_inches='tight')
# print('Figure {}_SD.png successfully created'.format(filename))
