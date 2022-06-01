# AII marker calculation


import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.linalg import eigh, eig
from numpy import pi
from random import seed
from functions import GaussianPointSet_3D, AmorphousHamiltonian3D_WT, local_marker_AII

start_time = time.time()
# %%  Global definitions

# Parameters of the model
n_orb = 4                   # Number of orbitals per site
n_neighbours = 6            # Number of neighbours
width = 0                   # Width of the gaussian for the WT model
density = 1                 # Point density of the RPS model
M = 2                       # Mass parameter in units of t1
t1, t2, lamb = 1, 0.1, 1    # Hopping and spin-orbit coupling in WT model
mu = 0                      # Disorder strength


# Lattice definition
L_x, L_y, L_z = 5, 5, 5                        # In units of a (average bond length)
n_sites = int(density * L_x * L_y * L_z)       # Number of sites in the lattice
n_states = n_sites * n_orb                     # Number of basis states
n_particles = int(n_states / 2)                # Number of filled states
sites = np.arange(0, L_x * L_y * L_z)          # Vector of sites
x = sites % L_x                                # x position in the crystalline case
y = (sites // L_x) % L_y                       # y position in the crystalline case
z = sites // (L_x * L_y)                       # z position in the crystalline case
x, y, z = GaussianPointSet_3D(x, y, z, width)  # Positions of the sites in the amorphous lattice


# Pauli matrices
sigma_0 = np.eye(2)                      # Pauli 0
sigma_x = np.array([[0, 1], [1, 0]])     # Pauli x
sigma_y = np.array([[0, -1j], [1j, 0]])  # Pauli y
sigma_z = np.array([[1, 0], [0, -1]])    # Pauli x
# %% Main

# Hamiltonian for the AII class
H_periodic, _ = AmorphousHamiltonian3D_WT(n_sites, n_orb, n_neighbours, L_x, L_y, L_z, x, y, z, M, t1, t2, lamb, "Closed")
energy_PBC, eigenstates_PBC = eigh(H_periodic)  # Eigenvalues, states
idx = energy_PBC.argsort()
energy_PBC = energy_PBC[idx]                    # Ordered energy eigenvalues
eigenstates_PBC = eigenstates_PBC[:, idx]       # Ordered eigenstates

# AII Projector
U = np.zeros((n_states, n_states), complex)  # Matrix for the AII projector
U[:, 0: n_particles] = eigenstates_PBC[:, 0: n_particles]
P = U @ np.conj(U.T)                         # AII projector

# Trivial Projector
R = np.zeros((n_states, n_states), complex)  # Matrix for the trivial projector
s = -np.kron(sigma_x, sigma_0)               # Chiral symmetry operator for the DIII class
for index in range(0, n_sites):
    R[index * n_orb: index * n_orb + n_orb, index * n_orb: index * n_orb + n_orb] = 0.5 * (np.eye(4) - s)


# DIII Projector
Q_aux1 = 1j * (P @ R - R @ P)  # Auxiliary operator
vals, vecs = eigh(Q_aux1)      # (real eigenvalues)
Q = 0.5 * vecs @ (np.diag(np.ones((n_states, )) - np.sign(vals))) @ np.conj(vecs.T)  # ( eigenvalues 0 and 1 projector)

# # Local Marker
list_sites = []
for index in range(L_x):
    aux = int(0.5 * (L_x * L_y * L_z) + 0.5 * (L_x * L_y) + index)  # Central array of sites
    list_sites.append(aux)                                          # Select the sites we want to calculate the marker on
marker = np.zeros((len(list_sites),))                               # Declaration of the local marker

for index in range(0, len(list_sites)):
    print(str(index) + "/" + str(len(list_sites)) + " S=sigma_y")
    marker[index] = local_marker_AII(n_orb, L_x, L_y, L_z, x, y, z, P, Q, R, list_sites[index])
print(np.mean(marker))


# Plot of the marker vs number of sites
plt.plot(range(len(list_sites)), marker, '.b')
plt.ylim(-3, 3)
plt.xlabel("Site number")
plt.ylabel("$\mathcal{C}$")
plt.text(2.5, 2, "$C=$" + str(np.mean(marker)))
plt.title("$L=$" + str(L_x) + "$, M=$" + str(M) + ", $t_2=$" + str(t2) + "$, S= \sigma_x$")
# plt.show()



