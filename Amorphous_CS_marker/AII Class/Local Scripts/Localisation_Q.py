# AII 3D Amorphous model: Seeing if Q is localised

import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from numpy import pi
from functions import GaussianPointSet_3D, AmorphousHamiltonian3D_WT, spectrum, displacement


start_time = time.time()
# %%  Global definitions

# Parameters of the model
n_orb = 4                    # Number of orbitals per site
n_neighbours = 6             # Number of neighbours
width = 0.1                  # Width of the gaussian for the WT model
density = 1                  # Point density of the RPS model
M = 2                        # Vector of mass parameters in units of t1
t1, t2, lamb = 1, 0.1, 1     # Hopping and spin-orbit coupling in WT model
mu = 0                       # Disorder strength
size = 8

# Pauli matrices
sigma_0 = np.eye(2)                         # Pauli 0
sigma_x = np.array([[0, 1], [1, 0]])        # Pauli x
sigma_y = np.array([[0, -1j], [1j, 0]])     # Pauli y
sigma_z = np.array([[1, 0], [0, -1]])       # Pauli z
tau_0, tau_x, tau_y, tau_z = sigma_0, sigma_x, sigma_y, sigma_z

# Global variables
L_x, L_y, L_z = None, None, None
n_sites, n_states, n_particles = None, None, None
sites, x, y, z = None, None, None, None

# Lattice definition
def lattice(size, n_orb, particles=None, chiral_symmetry=None, time_reversal=None):

    global L_x, L_y, L_z, n_sites, n_states, n_particles, sites, x, y, z, S, T

    L_x, L_y, L_z = size, size, size  # In units of a (average bond length)
    n_sites = L_x * L_y * L_z         # Number of sites in the lattice
    n_states = n_sites * n_orb        # Number of basis states

    if not particles:
        n_particles = int(n_states / 2)  # Half filling
    if chiral_symmetry is not None:
        S = np.zeros((n_states, n_states), complex)  # Chiral symmetry operator
        for index in range(0, n_sites):
            S[index * n_orb: index * n_orb + n_orb, index * n_orb: index * n_orb + n_orb] = chiral_symmetry
    if time_reversal is not None:
        T = np.zeros((n_states, n_states), complex)  # TR symmetry operator
        for index in range(0, n_sites):
            T[index * n_orb: index * n_orb + n_orb, index * n_orb: index * n_orb + n_orb] = time_reversal

    sites = np.arange(0, L_x * L_y * L_z)  # Vector of sites
    x = sites % L_x                        # x position in the crystalline case
    y = (sites // L_x) % L_y               # y position in the crystalline case
    z = sites // (L_x * L_y)               # z position in the crystalline case
lattice(size, n_orb)
x, y, z = GaussianPointSet_3D(x, y, z, width)  # Positions of the sites in the amorphous lattice


#%%  Main

# Hamiltonian for the AII class
H_periodic, _ = AmorphousHamiltonian3D_WT(n_sites, n_orb, n_neighbours, L_x, L_y, L_z, x, y, z, M, t1, t2, lamb, "Closed")
energy_PBC, eigenstates_PBC, P = spectrum(H_periodic, n_particles)

# Trivial Projector
R = np.zeros((n_states, n_states), complex)  # Matrix for the trivial projector
S = -np.kron(sigma_0, tau_y)                  # Chiral symmetry operator for the DIII class (in orbital basis only)
for index in range(0, n_sites):
    R[index * n_orb: index * n_orb + n_orb, index * n_orb: index * n_orb + n_orb] = 0.5 * (np.eye(4) - S)


# Spectrum of i[P, R]
H_aux = 1j * (P @ R - R @ P)    # Auxiliary operator
vals, vecs = eigh(H_aux)        # Spectrum
idx = vals.argsort()            # Ordered indexes
vals = vals[idx]                # Ordered energy eigenvalues
vecs = vecs[:, idx]             # Ordered eigenstates
Q = 0.5 * vecs @ (np.diag(np.ones((n_states,)) - np.sign(vals))) @ np.conj(vecs.T)  # ( eigenvalues 0 and 1 projector)


# Localisation properties of Q (we calculate along one direction for all sites, for example pick site 1, compare along x)
# Q1r = np.zeros([n_sites, ], complex)             # Declaration of Q(r-r')
# r_vec = np.zeros([n_sites, ], )                  # Declaration of r-r'
# psi_1[0], x1, y1, z1 = 1, x[0], y[0], z[0]       # State  |1, a, up> and position values
# for i in range(n_sites):
#     psi_r = np.zeros([n_states, ], complex)                   # State  |i, a, up>
#     psi_r[i * n_orb + 0], x2, y2, z2 = 1, x[i], y[i], z[i]    # State  |i, a, up> and position values
#     Q1r[i] = np.abs(np.conj(psi_1.T) @ Q @ psi_r)             # < 1 | Q | i >
#     r_vec[i], _, _ = displacement(x1, y1, z1, x2, y2, z2, L_x, L_y, L_z, "Closed")
#     print(Q1r[i])
Q1r = []
r_vec = []
for j in range(n_sites):
    psi_1 = np.zeros([n_states, ], complex)  # State  |1, a, up>
    psi_1[j * n_orb + 0], x1, y1, z1 = 1, x[j], y[j], z[j]  # State  |1, a, up> and position values
    for k in range(n_sites):
        r, _, _ = displacement(x1, y1, z1, x[k], y[k], z[k], L_x, L_y, L_z, "Closed")
        if r > (np.sqrt(3) * (size / 2) - 1):
            psi_r = np.zeros([n_states, ], complex)  # State  |i, a, up>
            psi_r[k * n_orb + 0], x2, y2, z2 = 1, x[k], y[k], z[k]  # State  |i, a, up> and position values
            Q1r.append(np.abs(np.abs(np.conj(psi_1.T) @ Q @ psi_r)))
            r_vec.append(r)
            # if np.abs(np.abs(np.conj(psi_1.T) @ Q @ psi_r)) > Q1r[i]:
            #     Q1r[i] = np.abs(np.abs(np.conj(psi_1.T) @ Q @ psi_r))  # < 1 | Q | i >
            #     print(j, k, r_vec, Q1r[i])

# Exponential fit
# fit = np.polyfit(r_vec, np.log(Q1r), 1)
# A, b = np.exp(fit[0]), fit[1]
# r_vec2 = np.linspace(0, max(r_vec), len(r_vec))
# loc_length = np.abs(np.real(1 / b))


# Plots
plt.plot(r_vec, np.abs(Q1r), '.b')
#plt.plot(np.linspace(0, max(r_vec)+0.1, 100), np.repeat(0, 100), '-.k', alpha=0.5)
# plt.plot(r_vec2, A * np.exp(b * r_vec2), '--r', linewidth=1)
# plt.text(2, 0.4, "$l=$" + '{:.2f}'.format(loc_length))
#plt.xlim(0, max(r_vec)+0.1)
#plt.ylim(-0.1, max(Q1r)+0.1)
plt.xlabel("$\\vert r - r' \\vert $")
plt.ylabel("$ Q(r - r') $")
plt.title("$L=$" + str(size) + " $M=$" + str(M) + " $t2=$" + str(t2) + " $w=$" + str(width) + " $S= \sigma_0 \otimes \sigma_y$")
plt.show()


end_time = time.time()
print("time elapsed=" + str(end_time - start_time) + "s")