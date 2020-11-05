import h5py
import include.vortex_detection

# Import datasets:
data_sqv = h5py.File('../../data/frames/100kf_1024nomag_psi_0=0.hdf5', 'r')
data_hqv = h5py.File('../../data/frames/100kf_1024nomag_newinit.hdf5', 'r')

# Grids:
x, y = data_sqv['grid/x'][...], data_sqv['grid/y'][...]

# Wavefunctions:
psi_plus_sqv = data_sqv['wavefunction/psi_plus']
psi_minus_sqv = data_sqv['wavefunction/psi_minus']
psi_plus_hqv = data_hqv['wavefunction/psi_plus']
psi_minus_hqv = data_hqv['wavefunction/psi_minus']

# Time data:
Nt, dt, Nframe = data_sqv['time/Nt'][:], data_sqv['time/dt'][:], data_sqv['time/Nframe'][:]

# Empty list for storing number of vortices:
sqv_number = []
hqv_number = []
num_of_frames = psi_plus_sqv.shape[-1]

# Detect and count vortices:
time_to_start = 10000   # Time in the simulation to start counting
start_frame = int(time_to_start // (dt * Nframe))
time_array = [time_to_start + i * dt for i in range(num_of_frames - start_frame)]

for i in range(start_frame, psi_plus_sqv.shape[-1]):
    _, _, _, _, _, _, vortex_number_sqv = include.vortex_detection.calculate_vortices(psi_plus_sqv[:, :, i],
                                                                                      psi_minus_sqv[:, :, i], x, y)
    sqv_number.append(vortex_number_sqv)

    _, _, _, _, _, _, vortex_number_hqv = include.vortex_detection.calculate_vortices(psi_plus_hqv[:, :, i],
                                                                                      psi_minus_hqv[:, :, i], x, y)
    hqv_number.append(vortex_number_hqv)

# Generate vortex file for saving data to:
vortex_file = h5py.File('vortex_data.hdf5', 'w')
vortex_file.create_dataset('sqv_number', data=sqv_number)
vortex_file.create_dataset('hqv_number', data=hqv_number)
