import h5py
import numpy as np
import include.diag as diag
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def angular_derivative(array, wvn):
    return np.fft.ifft(1j * wvn * np.fft.fft(array))


quench_rates = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850]
spin_winding_list = []

for quench in quench_rates:
    filename = '../../data/1d_kibble-zurek/1d_polar-BA-FM_{}.hdf5'.format(quench)
    with h5py.File(filename, 'r') as data_file:
        # Load in data:
        x = data_file['grid/x']
        Nx = len(x)
        dx = x[1] - x[0]
        dkx = 2 * np.pi / (Nx * dx)
        Kx = np.fft.fftshift(np.arange(-Nx // 2, Nx // 2) * dkx)

        dt = data_file['time/dt'][...]
        Nframe = data_file['time/Nframe'][...]
        frame = int(quench / (Nframe * dt))

        psi_plus = data_file['wavefunction/psi_plus'][:, frame]
        psi_0 = data_file['wavefunction/psi_0'][:, frame]
        psi_minus = data_file['wavefunction/psi_minus'][:, frame]
        n = abs(psi_plus) ** 2 + abs(psi_0) ** 2 + abs(psi_minus) ** 2

        # Calculate spin vectors:
        fx, fy, fz, F = diag.calculate_spin(psi_plus, psi_0, psi_minus, n)
        F_plus = fx + 1j * fy
        F_minus = fx - 1j * fy

        R = Nx * dx / (2 * np.pi)  # Radius of ring

        dF_plus = angular_derivative(F_plus, Kx)
        dF_minus = angular_derivative(F_minus, Kx)

        integral = (R / (2j * abs(F_plus) ** 2)) * (F_minus * dF_plus - F_plus * dF_minus)
        spin_winding = int(dx * sum(np.real(integral)) / (2 * np.pi * 2 * np.sqrt(Nx)))
        spin_winding_list.append(spin_winding)

        print('Spin winding for t_q={} is: {}'.format(quench, spin_winding))

plt.plot(quench_rates, spin_winding_list, 'ko')
plt.xlabel(r'$\tau_Q$')
plt.ylabel(r'$w$')
plt.show()
