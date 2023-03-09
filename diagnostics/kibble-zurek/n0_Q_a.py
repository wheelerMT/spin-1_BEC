import h5py
import numpy as np

quenches = [i for i in range(200, 1000, 100)]
runs = [i for i in range(1, 51)]
filename_prefix = "1d_BA-FM"

threshold = 0.01  # Percentage threshold difference


path = "../../scratch/data/spin-1/kibble-zurek"
diag_file = h5py.File(f"{path}/diagnostics/{filename_prefix}_n0_Q_a.hdf5", "w")
for quench in quenches:
    Q_a = []
    for run in runs:
        with h5py.File(
            f"{path}/ensembles/tau_q={quench}/{filename_prefix}_{run}.hdf5",
            "r",
        ) as data_file:
            print(f"Working on quench = {quench}, run {run}")

            # Grid data
            x = data_file["grid/x"]
            dx = x[1] - x[0]
            nx = len(x)

            psi_0 = data_file["wavefunction/psi_0"]
            time = data_file["time/t"][:, 0]
            Q = -time / quench

            analytical_n_0 = abs(0.5 * np.sqrt(2 + Q)) ** 2
            numerical_n_0 = (
                dx * np.sum(abs(psi_0[:, :]) ** 2, axis=0) / (nx * dx)
            )

            # Critical time index
            try:
                t_a_index = np.where(
                    numerical_n_0 - analytical_n_0 > threshold
                )[0][0]
                Q_a.append(abs(Q[t_a_index]))
            except IndexError:
                continue

    diag_file.create_dataset(f"{quench}/Q_a", data=Q_a)
