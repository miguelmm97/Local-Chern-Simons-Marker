# Plot for the lattice in an amorphous case

import numpy as np
from numpy import e, pi
import matplotlib.pyplot as plt
from random import gauss
from functions import GaussianPointSet_3D, AmorphousHamiltonian3D_WT

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

# Parameters of the model
size = 3
n_orb = 4                        # Number of orbitals per site
n_neighbours = 6                 # Number of neighbours
width = 0.1
M, t1, t2, lamb = 0, 0, 0, 0

# Global variables
L_x, L_y, L_z = None, None, None
n_sites, n_states, n_particles = None, None, None
sites, x, y, z = None, None, None, None


lattice(size, n_orb)  # Crystalline lattice definition, symmetries, ... etc
x, y, z = GaussianPointSet_3D(x, y, z, width)  # Positions of the sites in the amorphous lattice
H, neighbours = AmorphousHamiltonian3D_WT(n_sites, n_orb, n_neighbours, L_x, L_y, L_z, x, y, z, M, t1, t2, lamb, "Closed")

# cont = 0
# for index1 in range(0, n_sites):
#
#     if 0.5 < z[index1] < 1.5:
#
#         for index2 in range(n_neighbours):
#
#             if 0.5 < z[neighbours[index1, index2]] < 1.5:
#
#                 # plt.plot([x[index1], x[index2]], [y[index1], y[index2]], 'tab:gray', linewidth=1, alpha=0.3)
#                 plt.plot(x[index1], y[index1], '.b', markersize=10)  # Plot of the sites in the RPS
#                 plt.plot(x[index2], y[index2], '.b', markersize=10)  # Plot of the sites in the RPS
#
#     cont = cont + 1  # Update the counter so that we skip through index2 = previous indexes 1
#
# plt.xlim(-1, L_x)
# plt.ylim(-1, L_y)
# plt.xticks(color="w")
# plt.yticks(color="w")
# plt.show()


cont = 0
fig = plt.figure()
axes = fig.add_subplot(111, projection="3d")
# for index1 in range(0, n_sites):
#     for index2 in range(1, n_neighbours + 1):
#         axes.plot([x[index1], x[neighbours[index1, index2]]], [y[index1], y[neighbours[index1, index2]]],
#                   [z[index1], z[neighbours[index1, index2]]], 'k', linewidth=3, alpha=0.3)

axes.plot(x, y, z, '.b', markersize=10)  # Plot of the sites in the RPS
plt.show()