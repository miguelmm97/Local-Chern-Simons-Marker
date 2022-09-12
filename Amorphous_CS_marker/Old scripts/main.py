# DIII 3D Amorphous model: Gap/Marker as a function of the mass parameter

import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from numpy import pi
from random import seed
from functions import Lattice_graph, Hamiltonian3D, GaussianPointSet_3D,AmorphousHamiltonian3D_WT, local_marker

start_time = time.time()
# %%  Global definitions

# Parameters of the model
n_orb = 4  # Number of orbitals per site
n_neighbours = 6  # Number of neighbours
width = [0.2]  # Width of the gaussian for the WT model
R = 1.4  # Length cut-off for the hopping in RPS model
density = 1  # Point density of the RPS model
M = [2]  # np.linspace(-3.5, 3.5, 50)  # Vector of mass parameters in units of t1
t1, t2, lamb = 1, 0, 1  # Hopping and spin-orbit coupling in WT model
mu = 0  # Disorder strength

# Symmetries
sigma_0 = np.eye(2)  # Pauli 0
sigma_y = np.array([[0, -1j], [1j, 0]])  # Pauli y
S = - np.kron(sigma_0, sigma_y)  # Chiral symmetry operator for the WT model

# System sizes that we want to explore
size = [8]  # System size
size_x = np.repeat(size, len(width))  # Vector of system sizes in x direction
size_y = np.repeat(size, len(width))  # Vector of system sizes in y direction
size_z = np.repeat(size, len(width))  # Vector of system sizes in z direction

# Average over different realisations
# n_widths = len(width)
n_realisations = 1  # Realisations of the RPS to average over for each system size
n_sizes = len(size_x)  # Different system sizes for each block of realisations
gap_PBC = np.zeros([n_realisations, len(M)])  # Declaration of the vector containing the gaps PBC
gap_OBC = np.zeros([n_realisations, len(M)])  # Declaration of the vector containing the gaps PBC
gap_average_PBC = np.zeros([n_sizes, len(M)])  # Declaration of the gap average for each realisation PBC
gap_average_OBC = np.zeros([n_sizes, len(M)])  # Declaration of the gap average for each realisation OBC
marker_average = np.zeros([n_sizes, len(M)])  # Declaration of the local marker average
string_legend = []  # Strings for the legend of the plots
# for index in range(0, n_sizes): string_legend.append(str(size[index]))

# %% Periodic boundaries
# For M being just one value we calculate and plot the spectrum for the realisation of the lattice.
# For M being a vector of parameters we calculate some observable as a function of M

for num_size in range(0, n_sizes):
    for num_rea in range(0, n_realisations):

        # Lattice definition
        L_x, L_y, L_z = size_x[num_size], size_y[num_size], size_z[num_size]  # In units of a (average bond length)
        n_sites = int(density * L_x * L_y * L_z)  # Number of sites in the lattice
        n_states = n_sites * n_orb  # Number of basis states
        n_particles = int(n_states / 2)  # Number of filled states
        sites = np.arange(0, L_x * L_y * L_z)  # Vector of sites
        x = sites % L_x  # x position in the crystalline case
        y = (sites // L_x) % L_y  # y position in the crystalline case
        z = sites // (L_x * L_y)  # z position in the crystalline case
        x, y, z = GaussianPointSet_3D(x, y, z, width[num_size])  # Positions of the sites in the amorphous lattice

        # Declaration of the local marker
        list_sites = []
        for index in range(L_x):
            aux = int(0.5 * (L_x * L_y * L_z) + 0.5 * (L_x * L_y) + index)  # Central array of sites
            list_sites.append(aux)  # Select the sites we want to calculate the marker on
        marker = np.zeros((len(list_sites),))  # Declaration of the local marker

        # Hamiltonian
        for index in range(0, len(M)):
            print("size=" + str(num_size) + ", realisation=" + str(num_rea) + ", M=" + str(index))  # Sanity check
            H_periodic, matrix_neighbours = AmorphousHamiltonian3D_WT(n_sites, n_orb, n_neighbours, L_x, L_y, L_z, x, y,
                                                                                    z, M[index], t1, t2, lamb, "Closed")
            energy_PBC, eigenstates_PBC = eigh(H_periodic)
            idx = energy_PBC.argsort()
            energy_PBC = energy_PBC[idx]  # Ordered energy eigenvalues
            eigenstates_PBC = eigenstates_PBC[:, idx]  # Ordered eigenstates
            gap_PBC[num_rea, index] = energy_PBC[int(n_states / 2)]-energy_PBC[int(n_states / 2)-1]  # Energy gap

            # Local marker
            for site in range(len(list_sites)):
                marker[site] = local_marker(n_orb, n_sites, n_states, L_x, L_y, L_z, x, y, z,
                                                                 eigenstates_PBC, S, n_particles, list_sites[site])
            marker_average[num_size, index] = np.mean(marker)  # Average of the marker for the specified M

    gap_average_PBC[num_size, :] = np.mean(gap_PBC, 0)  # Averaged gap PBC

# Plots of the spectrum or the gap function (uncomment whatever you wanna plot)
    
# Band structure for a single realisation
plt.plot(range(0, n_states), energy_PBC, '.b', markersize=6)  # Plot of the energy
plt.ylim(-5, 5)
plt.xlim(0, n_states)
plt.xlabel("Eigenstate number")
plt.ylabel("Energy")
plt.legend(["PBC", "OBC"])
plt.title("density=" + str(density) + "   M=" + str(M))
plt.show()

# Marker as a function of the gap
for index in range(0, n_sizes):
    plt.plot(M, marker_average[index, :], '.')  # Plot of the gap PBC
plt.xlim(M[0], M[-1])
plt.xlabel("$M$")
plt.ylabel("$C marker$")
plt.legend(string_legend)
plt.title(" $N_{samples}=$" + str(n_realisations) + " $w=$" + str(width))
plt.show()

# Marker as a function of the width
# plt.plot(width, marker_average, '.-r')
# plt.plot(width, gap_average_PBC, '--b')
# plt.legend(("Marker", "$E_g$"))
# plt.xlabel("$w$")
# plt.ylabel("Marker, $E_g$")
# plt.show()

## Marker as a function of the lattice site
# plt.plot(range(len(list_sites)), marker, '.', markersize=6)
# plt.ylim(-3, 3)
# plt.xlabel('$site$')
# plt.ylabel('$Z$ local marker')
# plt.title("$N_x, N_y, N_z=$" + str(L_x))
# plt.show()


# %% Open Boundaries

# Main code
# for num_size in range(0, n_sizes):
#     for num_rea in range(0, n_realisations):
#
#         # Random point set (RPS) definition
#         L_x, L_y, L_z = size_x[num_size], size_y[num_size], size_z[num_size]  # In units of a (average bond length)
#         n_sites = int(density * L_x * L_y * L_z)  # Number of sites in the random point set
#         n_states = n_sites * n_orb  # Number of basis states
#         n_particles = int(n_states / 2)  # NUmber of filled states
#         sites = np.arange(0, L_x * L_y * L_z)  # Array with the number of each site
#         x = sites % L_x  # x position in the crystalline lattice
#         y = (sites // L_x) % L_y  # y position in the crystalline lattice
#         z = sites // (L_x * L_y)  # z position in the crystalline lattice
#         x, y, z = GaussianPointSet_3D(x, y, z, width[num_size])  # Positions of the sites in the amorphous lattice
#
#         # Open boundaries
#         for index in range(0, len(M)):
#             H_open, auxiliar = AmorphousHamiltonian3D_WT(n_sites, n_orb, n_neighbours, L_x, L_y, L_z, x, y, z,
#                                                                                     M[index], t1, t2, lamb, "Open")
#             energy_OBC, eigenstates_OBC = eigh(H_open)  # (points might not be ordered)
#             idx = energy_OBC.argsort()  # [:n] #add [:n] to only pick out the n lowest eigenvalues.
#             energy_OBC = energy_OBC[idx]  # Ordered energy eigenvalues
#             eigenstates_OBC = eigenstates_OBC[:, idx]  # Ordered eigenstates
#             gap_OBC[num_rea, index] = energy_OBC[int(n_states / 2)] - energy_OBC[int(n_states / 2) - 1]  # Energy gap
#
#     gap_average_OBC[num_size, :] = np.mean(gap_OBC, 0)  # Averaged gap OBC

# # PBC and OBC band structure
# plt.plot(range(0, n_states), energy_PBC, '.b', markersize=6)  # Plot of the energy
# plt.plot(range(0, n_states), energy_OBC, '.r', markersize=3)  # Plot of the energy
# plt.ylim(-5, 5)
# plt.xlim(0, n_states)
# plt.xlabel("Eigenstate number")
# plt.ylabel("Energy")
# plt.legend(["PBC", "OBC"])
# plt.title("density=" + str(density) + "   M=" + str(M))
# plt.show()


# %%  Plot of the RPS

# Lattice_graph(L_x, L_y, L_z, n_sites, n_neighbours, x, y, z, matrix_neighbours)

# %% Crystalline limit


# for num_size in range(0, n_sizes):
#     for num_rea in range(0, n_realisations):
#         # Random point set (RPS) definition
#         L_x, L_y, L_z = size_x[num_size], size_y[num_size], size_z[num_size]  # In units of a (average bond length)
#         n_sites = int(density * L_x * L_y * L_z)  # Number of sites in the random point set
#         n_states = n_sites * n_orb  # Number of basis states
#         n_particles = int(n_states / 2)
#         sites = np.arange(0, L_x * L_y * L_z)  # Array with the number of each site
#         x = sites % L_x  # x position
#         y = (sites // L_x) % L_y  # y position
#         z = sites // (L_x * L_y)  # z position
#         list_sites = []
#         for index in range(L_x):
#             aux = int(0.5 * (L_x * L_y * L_z) + 0.5 * (L_x * L_y) + index)
#             list_sites.append(aux)
          #list_sites = np.arange(n_sites)
          # Periodic boundaries
          # for index in range(0, len(M)):
          #     print("size=" + str(num_size) + ", realisation=" + str(num_rea) + ", M=" + str(index))  # Sanity check
          #     H_periodic = Hamiltonian3D(L_x, L_y, L_z, n_orb, sites, x, y, z, M[index], t1, t2, lamb, mu, "Closed")
          #     energy_PBC, eigenstates_PBC = eigh(H_periodic)
          #     idx = energy_PBC.argsort()
          #     energy_PBC = energy_PBC[idx]  # Ordered energy eigenvalues
          #     eigenstates_PBC = eigenstates_PBC[:, idx]
          #     gap_PBC[num_rea, index] = energy_PBC[int(n_states / 2)]-energy_PBC[int(n_states / 2)-1]

    #     # Open boundaries
    #     for index in range(0, len(M)):
    #         H_open = Hamiltonian3D(L_x, L_y, L_z, n_orb, sites, x, y, z, M[index], t1, t2, lamb, mu, "Open")
    #         energy_OBC, eigenstates_OBC = eigh(H_open)  # (points might not be ordered)
    #         idx = energy_OBC.argsort()  # [:n] #add [:n] to only pick out the n lowest eigenvalues.
    #         energy_OBC = energy_OBC[idx]  # Ordered energy eigenvalues
    #         eigenstates_OBC = eigenstates_OBC[:, idx]
    #         gap_OBC[num_rea, index] = energy_OBC[int(n_states / 2)] - energy_OBC[int(n_states / 2) - 1]
    #         CS_marker = CS_local_marker(n_orb, n_sites, n_states, x, y, z, eigenstates_OBC, S, n_particles, list_sites)
    # #
    # gap_average_PBC[num_size, :] = np.mean(gap_PBC, 0)  # Averaged gap PBC
    # gap_average_OBC[num_size, :] = np.mean(gap_OBC, 0)  # Averaged gap OBC


# Plots of the spectrum or the gap function
# if len(M) == 1 and n_realisations == 1 and n_sizes == 1:
#     plt.plot(range(0, n_states), energy_PBC, '.b', markersize=6)  # Plot of the energy
#     plt.plot(range(0, n_states), energy_OBC, '.r', markersize=3)  # Plot of the energy
#     # plt.ylim(-5, 5)
#     plt.xlim(0, n_states)
#     plt.xlabel("Eigenstate number")
#     plt.ylabel("Energy")
#     plt.legend(["PBC", "OBC"])
#     plt.title("density=" + str(density) + "   M=" + str(M))
#     plt.show()
# elif len(M) > 1:
#     for index in range(0, n_sizes):
#         plt.plot(M, gap_average_PBC[index, :], '.', markersize=6)  # Plot of the gap PBC
#         # plt.plot(M, gap_average_OBC[index, :], '.r', markersize=3)  # Plot of the gap OBC
#     plt.xlim(M[0], M[-1])
#     plt.xlabel("$m$")
#     plt.ylabel("$E_g$")
#     plt.legend(string_legend)
#     plt.title(" $N_{samples}=$" + str(n_realisations))
#     plt.show()


# plt.plot(range(len(list_sites)), CS_marker, '.', markersize=6)
# plt.ylim(-3, 3)
# plt.xlabel('$site$')
# plt.ylabel('$Z$ local marker')
# plt.title("$N_x, N_y, N_z=$" + str(L_x))
# plt.show()

# plt.savefig("marker2_s10_mu05.pdf", bbox_inches="tight")


# %%
end_time = time.time()
print("time elapsed=" + str(end_time - start_time) + "s")




















# sites = np.arange(0, L_x * L_y * L_z)  # Array with the number of each site
# x = sites % L_x  # x position
# y = (sites // L_x) % L_y  # y position
# z = sites // (L_x * L_y)  # z position


# # Periodic boundaries
# for index in range(0, len(M)):
#     print("size=" + str(num_size) + ", realisation=" + str(num_rea) + ", M=" + str(index))  # Sanity check
#     H_periodic = AmorphousHamiltonian3D(n_sites, n_orb, L_x, L_y, L_z, x, y, z, M[index], t1, t2, lamb, R, "Closed")
#     # H_periodic = Hamiltonian3D(L_x, L_y, L_z, n_orb, sites, x, y, z, M[index], t1, t2, lamb, "Closed")
#     energy_PBC, eigenstates_PBC = eigh(H_periodic)
#     idx = energy_PBC.argsort()
#     energy_PBC = energy_PBC[idx]  # Ordered energy eigenvalues
#     eigenstates_PBC = eigenstates_PBC[:, idx]
#     gap_PBC[num_rea, index] = energy_PBC[int(n_states / 2)] - energy_PBC[int(n_states / 2) - 1]

# Open boundaries
# for index in range(0, len(M)):
#     H_open = AmorphousHamiltonian3D(n_sites, n_orb, L_x, L_y, L_z, x, y, z, M[index], t1, t2, lamb, R, "Open")
#     # H_open = Hamiltonian3D(L_x, L_y, L_z, n_orb, sites, x, y, z, M[index], t1, t2, lamb, "Open")
#     energy_OBC, eigenstates_OBC = eigh(H_open)  # (points might not be ordered)
#     idx = energy_OBC.argsort()  # [:n] #add [:n] to only pick out the n lowest eigenvalues.
#     energy_OBC = energy_OBC[idx]  # Ordered energy eigenvalues
#     eigenstates_OBC = eigenstates_OBC[:, idx]
#     gap_OBC[num_rea, index] = energy_OBC[int(n_states / 2)] - energy_OBC[int(n_states / 2) - 1]