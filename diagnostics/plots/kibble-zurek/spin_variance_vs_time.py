import numpy as np
import h5py
import include.diag as diag
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def angular_derivative(array, wvn):
    return np.fft.ifft(1j * wvn * np.fft.fft(array))


# Load in data:
filename = '1d_kibble-zurek/1d_polar-BA-FM_500'  # input('Enter data filename: ')
data_file = h5py.File('../../../data/{}.hdf5'.format(filename), 'r')

# Create necessary grid and time data
runs = [i for i in range(1, 11)]
dt = data_file['time/dt'][...]
Nframe = data_file['time/Nframe'][...]
time = dt * Nframe * np.arange(1, 200 + 1)
x = data_file['grid/x']
Nx = len(x)
dx = x[1] - x[0]
R = Nx * dx / (2 * np.pi)  # Radius of ring
dkx = 2 * np.pi / (Nx * dx)
Kx = np.fft.fftshift(np.arange(-Nx // 2, Nx // 2) * dkx)

quench_rate = 500
frame = int(quench_rate / (Nframe * dt))

spin_winding = np.empty((10, 200))
for run in runs:
    for i in range(200):
        psi_plus = data_file['run{}/wavefunction/psi_plus'.format(run)][:, i]
        psi_0 = data_file['run{}/wavefunction/psi_0'.format(run)][:, i]
        psi_minus = data_file['run{}/wavefunction/psi_minus'.format(run)][:, i]
        n = abs(psi_plus) ** 2 + abs(psi_0) ** 2 + abs(psi_minus) ** 2

        # Calculate spin vectors:
        fx, fy, fz, F = diag.calculate_spin(psi_plus, psi_0, psi_minus, n)
        F_plus = fx + 1j * fy
        F_minus = fx - 1j * fy

        dF_plus = angular_derivative(F_plus, Kx)
        dF_minus = angular_derivative(F_minus, Kx)

        integral = (R / (2j * abs(F_plus) ** 2)) * (F_minus * dF_plus - F_plus * dF_minus)
        spin_winding[run - 1, i] = int(dx * sum(np.real(integral)) / (2 * np.pi * 2 * np.sqrt(Nx)))

# Need to calculate variance:
spin_variance = np.empty(200, )
for i in range(200):
    variance_list = []
    for run in runs:
        variance_list.append(spin_winding[run - 1, i])
    spin_variance[i] = np.var(variance_list)

plt.plot(time[:100], spin_variance[:100], 'k')
plt.xlabel(r'$t/\tau$')
plt.ylabel(r'<$w^2>$')
plt.ylim(0, 20)
plt.tight_layout()
plt.show()
