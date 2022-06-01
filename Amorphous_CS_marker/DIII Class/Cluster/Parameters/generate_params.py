# Generate parameter data to submit to the cluster
import numpy as np
import h5py

Ls = [8, 10, 12]  # System sizes
Ms = np.linspace(-3.5, 3.5, 70)  # Mass parameter
Ws = np.linspace(0.2, 0.3, 25)
Rs = np.arange(4)  # Realisations
Ms_inset = [0, 2, 4]
Ms_width = [0, 2, 4]


# with h5py.File('params_M.h5', 'w') as f:
#      f.create_dataset("mytestdata", data=[Ls.T, Ms.T, Rs.T])

with open('params_M.txt', 'w') as f:
    for L in Ls:
        for M in Ms:
            for R in Rs:
                f.write('{} {:.4f} {}\n'.format(L, M, R))

with open('M_values.txt', 'w') as f:
    for M in Ms:
        f.write('{:.4f}\n'.format(M))


with open('params_M_inset.txt', 'w') as f:
    for M in Ms_inset:
        for R in Rs:
            f.write('{} {}\n'.format(M, R))


with open('params_width.txt', 'w') as f:
    for M in Ms_width:
        for L in Ls:
            for W in Ws:
                for R in Rs:
                    f.write('{} {} {:.8f} {}\n'.format(M, L, W, R))

with open('width_values.txt', 'w') as f:
    for W in Ws:
        f.write('{:.8f}\n'.format(W))


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