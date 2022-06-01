# Generate parameter data to submit to the cluster
import numpy as np
import h5py

Ls = [10]  # System sizes
Ms = [0, 2]  # Mass parameter
Ws = [0.025, 0.05, 0.1, 0.2]
Ts = np.linspace(0, 2, 20)  # Chiral symmetry breaking
Rs = np.arange(3)  # Realisations


# with h5py.File('params_M.h5', 'w') as f:
#      f.create_dataset("mytestdata", data=[Ls.T, Ms.T, Rs.T])

with open('params_t2.txt', 'w') as f:
    for L in Ls:
        for M in Ms:
            for W in Ws:
                for T in Ts:
                    for R in Rs:
                        f.write('{} {} {:.8f} {:.4f} {}\n'.format(L, M, W, T, R))

with open('t2_values.txt', 'w') as f:
    for W in Ws:
        for T in Ts:
            f.write('{:.8f} {:.4f} \n'.format(W, T))



