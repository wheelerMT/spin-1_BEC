import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
import h5py
import time


# Load the data:
filename = 'scalar'
data_path = '../../data/{}.hdf5'.format(filename)

data_file = h5py.File(data_path, 'r')

# Loading grid array data:
x, y = data_file['grid/x'], data_file['grid/y']
X, Y = np.meshgrid(x[:], y[:])
Nx, Ny = x[:].size, y[:].size
dx, dy = x[1] - x[0], y[1] - y[0]

# Load in time data:
Nt, dt, Nframe = data_file['time/Nt'][...], data_file['time/dt'][...], data_file['time/Nframe'][...]

# Load in wavefunction:
initial_state = data_file['initial_state/psi']
psi = data_file['wavefunction/psi']
num_of_frames = psi.shape[-1]

# Determine contour values:
cvals = np.linspace(0, np.max(abs(initial_state[:, :]) ** 2), 50, endpoint=True)

# Generate the figure:
fig, ax = plt.subplots()
ax.set_xlabel(r'$x / \xi$')
ax.set_ylabel(r'$y / \xi$')

contour = [ax.contourf(X, Y, abs(psi[:, :, 0]) ** 2, cvals, cmap='gnuplot')]


# Function to update plot:
def update_plot(i):
    t1 = time.time()
    for tp in contour[0].collections:
        tp.remove()
    contour[0] = ax.contourf(X, Y, abs(psi[:, :, i]) ** 2, cvals, cmap='gnuplot')
    plt.suptitle(r'$\tau = {:.2f}$'.format(i * dt * Nframe))
    print('On plot {}. The time for this plot was {:.2f}s'.format(i, time.time()-t1))
    return contour[0].collections


anim = matplotlib.animation.FuncAnimation(fig, update_plot, frames=num_of_frames,
                                          interval=10, blit=False, repeat=True)
anim.save('scalar_turbulence.mp4', dpi=400,
          writer=matplotlib.animation.FFMpegWriter(fps=60))

