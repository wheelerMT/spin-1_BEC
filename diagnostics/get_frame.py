import h5py
import numpy as np


scalar_or_spinor = input('Which type of simulation are we working with? (spinor/scalar): ')

if scalar_or_spinor == 'spinor':
    # Opening main data set:
    filename = input('Enter filename: ')
    data_path = '../polar_phase/{}_data.hdf5'.format(filename)

    file = h5py.File(data_path, 'r')

    # Opening useful data to re-save to frame data:
    x, y = file['grid/x'], file['grid/y']
    Nx = len(x)
    Ny = len(y)
    dt = file['time/dt'][...]
    Nframe = file['time/Nframe'][...]

    # Loading wavefunction
    psi_plus = file['wavefunction/psi_plus']
    psi_0 = file['wavefunction/psi_0']
    psi_minus = file['wavefunction/psi_minus']

    # Number of frames of data in main file:
    num_of_frames = np.ma.size(psi_0[0, 0, :], -1)
    print('Number of frames = %i' % num_of_frames)

    # Gets the amount of frames and the corresponding times from user:
    num_of_saved_frames = int(input('Enter the number of frames you wish to save: '))
    saved_time = []
    for i in range(num_of_saved_frames):
        frame_time = input('Enter the time of a frame you wish to save ({} / {}): '. format(i + 1, num_of_saved_frames))
        saved_time.append(int(frame_time))
    saved_time.sort()

    # Path to save data frame:
    frame_path = 'polar_phase/{}_{}.hdf5'.format(filename, saved_time[-1])

    # Creates the frame dataset:
    with h5py.File(frame_path, 'w') as frame:
        frame.create_dataset('grid/x', data=x), frame.create_dataset('grid/y', data=y)
        frame.create_dataset('num_of_frames', data=num_of_frames)
        psi_plus_frame = frame.create_dataset('wavefunction/psi_plus', (Nx, Ny, num_of_saved_frames),
                                              dtype='complex64')
        psi_0_frame = frame.create_dataset('wavefunction/psi_0', (Nx, Ny, num_of_saved_frames), dtype='complex64')
        psi_minus_frame = frame.create_dataset('wavefunction/psi_minus', (Nx, Ny, num_of_saved_frames),
                                               dtype='complex64')

        # Saves appropriate frames:
        for i in range(num_of_saved_frames):
            if saved_time[i] == 0:  # Ensures correct array position when we want t = 0
                psi_plus_frame[:, :, i] = psi_plus[:, :, 0]
                psi_0_frame[:, :, i] = psi_0[:, :, 0]
                psi_minus_frame[:, :, i] = psi_minus[:, :, 0]
            else:
                psi_plus_frame[:, :, i] = psi_plus[:, :, saved_time[i] // int(Nframe * dt) - 1]
                psi_0_frame[:, :, i] = psi_0[:, :, saved_time[i] // int(Nframe * dt) - 1]
                psi_minus_frame[:, :, i] = psi_minus[:, :, saved_time[i] // int(Nframe * dt) - 1]

    file.close()

if scalar_or_spinor == 'scalar':
    # Opening main dataset:
    filename = 'scalar'
    data_path = '../scalar/{}_data.hdf5'.format(filename)
    file = h5py.File(data_path, 'r')

    # Opening useful data to re-save to frame data:
    x, y = file['grid/x'], file['grid/y']
    Nx = len(x)
    Ny = len(y)
    dt = file['time/dt'][...]
    Nframe = file['time/Nframe'][...]

    # Loading wavefunction:
    psi = file['wavefunction/psi']

    # Number of frames of data in main file:
    num_of_frames = np.ma.size(psi[0, 0, :], -1)
    print('Number of frames = %i' % num_of_frames)

    # Gets the amount of frames and the corresponding times from user:
    num_of_saved_frames = int(input('Enter the number of frames you wish to save: '))
    saved_time = []
    for i in range(num_of_saved_frames):
        frame_time = input('Enter the time of a frame you wish to save ({} / {}): '.format(i + 1, num_of_saved_frames))
        saved_time.append(int(frame_time))
    saved_time.sort()

    # Path to save data frame:
    frame_path = 'scalar/{}_{}.hdf5'.format(filename, saved_time[-1])

    # Creates the frame dataset:
    with h5py.File(frame_path, 'w') as frame:
        frame.create_dataset('grid/x', data=x), frame.create_dataset('grid/y', data=y)
        frame.create_dataset('num_of_frames', data=num_of_frames)
        psi_frame = frame.create_dataset('wavefunction/psi', (Nx, Ny, 3), dtype='complex64')

        # Saves appropriate frames:
        for i in range(num_of_saved_frames):
            if saved_time[i] == 0:  # Ensures correct array position when we want t = 0
                psi_frame[:, :, i] = psi[:, :, 0]
            else:
                psi_frame[:, :, i] = psi[:, :, saved_time[i] // int(Nframe * dt) - 1]
    file.close()
