# DIII 3D Amorphous model: Band structure for closed and open boundaries

import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from numpy import pi
from random import seed
from functions import Lattice_graph, Hamiltonian3D, GaussianPointSet_3D, AmorphousHamiltonian3D_WT, local_marker

start_time = time.time()
# %%  Global definitions

# Parameters of the model
n_orb = 4               # Number of orbitals per site
n_neighbours = 6        # Number of neighbours
width = 0.2             # Width of the gaussian for the WT model
density = 1             # Point density of the RPS model
M = 2                   # Mass parameter in units of t1
t1, t2, lamb = 1, 0, 1  # Hopping and spin-orbit coupling in WT model
mu = 0                  # Disorder strength


# Lattice definition
L_x, L_y, L_z = 8, 8, 8                                  # In units of a (average bond length)
n_sites = int(density * L_x * L_y * L_z)                 # Number of sites in the lattice
n_states = n_sites * n_orb                               # Number of basis states
n_particles = int(n_states / 2)                          # Number of filled states
sites = np.arange(0, L_x * L_y * L_z)                    # Vector of sites
x = sites % L_x                                          # x position in the crystalline case
y = (sites // L_x) % L_y                                 # y position in the crystalline case
z = sites // (L_x * L_y)                                 # z position in the crystalline case
x, y, z = GaussianPointSet_3D(x, y, z, width[num_size])  # Positions of the sites in the amorphous lattice

# %% Main
    
# Hamiltonians
H_periodic, matrix_neighbours = AmorphousHamiltonian3D_WT(n_sites, n_orb, n_neighbours, L_x, L_y, L_z, x, y,
                                                                                          z, M, t1, t2, lamb, "Closed")
H_open, aux = AmorphousHamiltonian3D_WT(n_sites, n_orb, n_neighbours, L_x, L_y, L_z, x, y, z,
                                                                                           M, t1, t2, lamb, "Open")
# PBC
energy_PBC, eigenstates_PBC = eigh(H_periodic)  # (points might not be ordered)
idx = energy_PBC.argsort()                      # [:n] #add [:n] to only pick out the n lowest eigenvalues.
energy_PBC = energy_PBC[idx]                    # Ordered energy eigenvalues
eigenstates_PBC = eigenstates_PBC[:, idx]       # Ordered eigenstates
# OBC
energy_OBC, eigenstates_OBC = eigh(H_open)  # (points might not be ordered)
idx = energy_OBC.argsort()                  # [:n] #add [:n] to only pick out the n lowest eigenvalues.
energy_OBC = energy_OBC[idx]                # Ordered energy eigenvalues
eigenstates_OBC = eigenstates_OBC[:, idx]   # Ordered eigenstates


# Band structure for a single realisation
plt.plot(range(0, n_states), energy_PBC, '.b', markersize=6)  # Plot of the energy
plt.plot(range(0, n_states), energy_OBC, '.r', markersize=6)  # Plot of the energy
plt.ylim(-5, 5)
plt.xlim(0, n_states)
plt.xlabel("Eigenstate number")
plt.ylabel("Energy")
plt.legend(["PBC", "OBC"])
plt.title(" M=" + str(M))
plt.show()
