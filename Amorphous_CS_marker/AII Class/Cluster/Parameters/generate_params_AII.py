# Generate parameter data to submit to the cluster
import numpy as np
import h5py

Ls = [8, 10, 12]  # System sizes
Ms = [2]  # Mass parameter
Ws = np.linspace(0, 0.2, 50)
Ts = [0.5]  # Chiral symmetry breaking
Rs = np.arange(4)  # Realisations


# with h5py.File('params_M.h5', 'w') as f:
#      f.create_dataset("mytestdata", data=[Ls.T, Ms.T, Rs.T])

with open('params_width.txt', 'w') as f:
    for L in Ls:
        for M in Ms:
            for W in Ws:
                for T in Ts:
                    for R in Rs:
                        f.write('{} {} {:.8f} {:.4f} {}\n'.format(L, M, W, T, R))

with open('width_values.txt', 'w') as f:
    for W in Ws:
            f.write('{:.8f} \n'.format(W))

with open('params_pd.txt', 'w') as f:
    for M in Ms:
        for W in Ws:
            for R in Rs:
                f.write('{:.4f} {:.8f} {}\n'.format(M, W, R))

with open('PD_values_W.txt', 'w') as f:
    for W in Ws:
        f.write('{:.8f}\n'.format(W))

with open('PD_values_M.txt', 'w') as f:
    for M in Ms:
        f.write('{:.4f}\n'.format(M))