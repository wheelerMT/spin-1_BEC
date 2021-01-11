import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
import h5py
import time

# Load the data:
filename = 'frames/100kf_scalar'
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
psi = data_file['wavefunction/psi']
num_of_frames = psi.shape[-1]

# Determine contour values:
cvals = np.linspace(0, 2000, 25, endpoint=True)

# Generate the figure:
fig, ax = plt.subplots(1, figsize=(8, 6))
ax.set_xlabel(r'$x / \xi$')
ax.set_ylabel(r'$y / \xi$')
ax.set_xlim(x[:].min(), x[:].max())
ax.set_ylim(y[:].min(), y[:].max())


# Initial plot:
contour = [ax.contourf(X, Y, abs(psi[:, :, 0]) ** 2, cvals, cmap='gnuplot')]
plt.colorbar(contour[0], ax=ax)


def update_plot(i):
    global contour

    for c in contour[0].collections:
        c.remove()

    t1 = time.time()
    contour[0] = ax.contourf(X, Y, abs(psi[:, :, i]) ** 2, cvals, cmap='gnuplot')
    plt.suptitle(r'$\tau = {:.2f}$'.format(i * dt * Nframe))
    print('On plot {}. The time for this plot was {:.2f}s'.format(i, time.time() - t1))
    return contour[0].collections


anim = matplotlib.animation.FuncAnimation(fig, update_plot, frames=num_of_frames)
anim.save('{}.mp4'.format(filename), writer=matplotlib.animation.FFMpegWriter(fps=60))
