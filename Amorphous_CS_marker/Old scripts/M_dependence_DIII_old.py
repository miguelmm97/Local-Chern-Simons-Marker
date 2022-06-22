# DIII 3D Amorphous model: Gap/Marker as a function of the mass parameter

import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.linalg import eigh
import matplotlib.ticker as ticker
from numpy import pi
from functions import GaussianPointSet_3D, AmorphousHamiltonian3D_WT, local_marker_DIII

start_time = time.time()
# %%  Global definitions

# Parameters of the model
n_orb = 4  # Number of orbitals per site
n_neighbours = 6  # Number of neighbours
width = 0.1  # Width of the gaussian for the WT model
density = 1  # Point density of the RPS model
M = np.linspace(-3.8, 3.8, 30)  # Vector of mass parameters in units of t1
M_inset = [0, 2, 4]  # Values of M for the inset figure
t1, t2, lamb = 1, 0, 1  # Hopping and spin-orbit coupling in WT model
mu = 0  # Disorder strength
# R = 1.4  # Length cut-off for the hopping in RPS model
size = [3]  # System size

# Symmetries
sigma_0 = np.eye(2)  # Pauli 0
sigma_y = np.array([[0, -1j], [1j, 0]])  # Pauli y
s = - np.kron(sigma_0, sigma_y)  # Chiral symmetry operator for the WT model

# Definition of global variables
L_x, L_y, L_z = None, None, None
n_sites, n_states, n_particles = None, None, None
sites, x, y, z = None, None, None, None
S = None

# Average over different realisations
n_realisations = 1  # Realisations of the RPS to average over for each system size
n_sizes = len(size)  # Different system sizes for each block of realisations
gap_PBC = np.zeros([n_realisations, len(M)])  # Declaration of the vector containing the gaps PBC
gap_average_PBC = np.zeros([n_sizes, len(M)])  # Declaration of the gap average for each realisation PBC
marker_final = np.zeros((n_sizes, len(M)))  # Declaration of the marker average
marker_inset = np.zeros((len(M_inset), size[-1]))  # Declaration of the marker average for the inset figure


def lattice(size, n_orb, particles=None, chiral_symmetry=None, time_reversal=None):
    global L_x, L_y, L_z, n_sites, n_states, n_particles, sites, x, y, z, S, T

    L_x, L_y, L_z = size, size, size  # In units of a (average bond length)
    n_sites = L_x * L_y * L_z  # Number of sites in the lattice
    n_states = n_sites * n_orb  # Number of basis states

    if not particles:
        n_particles = int(n_states / 2)  # Number of filled states
    if chiral_symmetry is not None:
        S = np.zeros((n_states, n_states), complex)  # Chiral symmetry operator
        for index in range(0, n_sites):
            S[index * n_orb: index * n_orb + n_orb, index * n_orb: index * n_orb + n_orb] = chiral_symmetry
    if time_reversal is not None:
        T = np.zeros((n_states, n_states), complex)  # Chiral symmetry operator
        for index in range(0, n_sites):
            T[index * n_orb: index * n_orb + n_orb, index * n_orb: index * n_orb + n_orb] = time_reversal

    sites = np.arange(0, L_x * L_y * L_z)  # Vector of sites
    x = sites % L_x  # x position in the crystalline case
    y = (sites // L_x) % L_y  # y position in the crystalline case
    z = sites // (L_x * L_y)  # z position in the crystalline case



# %% Main

# Calculation of the marker with M (main figure)
# Iterate number of sites
for num_size in range(0, n_sizes):

    # Iterate number of realisations
    for num_rea in range(0, n_realisations):
        realisation_time = time.time()
        print("time realisation = " + str(realisation_time - start_time))

        # Lattice definition
        # L_x, L_y, L_z = size_x[num_size], size_y[num_size], size_z[num_size]  # In units of a (average bond length)
        # n_sites = L_x * L_y * L_z  # Number of sites in the lattice
        # n_states = n_sites * n_orb  # Number of basis states
        # n_particles = int(n_states / 2)  # Number of filled states
        # sites = np.arange(0, L_x * L_y * L_z)  # Vector of sites
        # x = sites % L_x  # x position in the crystalline case
        # y = (sites // L_x) % L_y  # y position in the crystalline case
        # z = sites // (L_x * L_y)  # z position in the crystalline case
        lattice(size[num_size], n_orb, chiral_symmetry=s)
        print(L_x)
        x, y, z = GaussianPointSet_3D(x, y, z, width)  # Positions of the sites in the amorphous lattice

        # Chiral symmetry
        # S = np.zeros((n_states, n_states), complex)  # Chiral symmetry operator
        # for index in range(0, n_sites):
        #     S[index * n_orb: index * n_orb + n_orb, index * n_orb: index * n_orb + n_orb] = s

        # Declaration of the local marker
        list_sites = []
        for index in range(L_x):
            aux = int(0.5 * (L_x * L_y * L_z) + 0.5 * (L_x * L_y) + index)  # Central array of sites
            list_sites.append(aux)  # Select the sites we want to calculate the marker on
        marker = np.zeros((len(list_sites),))  # Declaration of the local marker
        marker_average = np.zeros((len(M), ))  # Declaration of the marker average  over sites


        # Iterate in the mass parameter
        for index in range(0, len(M)):

            print("size=" + str(num_size) + ", realisation=" + str(num_rea) + ", M=" + str(index))  # Sanity check

            # Hamiltonian
            H_periodic, matrix_neighbours = AmorphousHamiltonian3D_WT(n_sites, n_orb, n_neighbours, L_x, L_y, L_z, x, y,
                        z, M[index], t1, t2, lamb, "Closed")
            energy_PBC, eigenstates_PBC = eigh(H_periodic)
            idx = energy_PBC.argsort()
            energy_PBC = energy_PBC[idx]  # Ordered energy eigenvalues
            eigenstates_PBC = eigenstates_PBC[:, idx]  # Ordered eigenstates
            # gap_PBC[num_rea, index] = energy_PBC[int(n_states / 2)] - energy_PBC[int(n_states / 2) - 1]  # Energy gap

            # Valence band projector
            U = np.zeros((n_states, n_states), complex)  # Matrix for the projector
            U[:, 0: n_particles] = eigenstates_PBC[:, 0: n_particles]
            P = U @ np.conj(np.transpose(U))  # Projector onto the occupied subspace

            # Local marker calculation (amorphous average for free in just one realisation)
            for site in range(len(list_sites)):
                marker[site] = local_marker_DIII(n_orb, L_x, L_y, L_z, x, y, z, P, S, list_sites[site])
            marker_average[index] = np.mean(marker)  # Average of the marker for the specified M over sites


        marker_final[num_size, :] = marker_final[num_size, :] + marker_average  # Cumulative marker

    marker_final[num_size, :] = marker_final[num_size, :] / n_realisations  # Average of the marker over different realisations
    # gap_average_PBC[num_size, :] = np.mean(gap_PBC, 0)  # Averaged gap PBC


# Calculation of the marker as a function of the sites (Inset figure)
# Lattice definition
lattice(size[-1], n_orb, chiral_symmetry=s)
# L_x, L_y, L_z = size_x[-1], size_y[-1], size_z[-1]  # In units of a (average bond length)
# n_sites = int(density * L_x * L_y * L_z)  # Number of sites in the lattice
# n_states = n_sites * n_orb  # Number of basis states
# n_particles = int(n_states / 2)  # Number of filled states
# sites = np.arange(0, L_x * L_y * L_z)  # Vector of sites
# x = sites % L_x  # x position in the crystalline case
# y = (sites // L_x) % L_y  # y position in the crystalline case
# z = sites // (L_x * L_y)  # z position in the crystalline case
#
# # Chiral symmetry
# S = np.zeros((n_states, n_states), complex)  # Chiral symmetry operator
# for index in range(0, n_sites):
#     S[index * n_orb: index * n_orb + n_orb, index * n_orb: index * n_orb + n_orb] = s

# Declaration of the local marker
list_sites = []
for index in range(L_x):
    aux = int(0.5 * (L_x * L_y * L_z) + 0.5 * (L_x * L_y) + index)  # Central array of sites
    list_sites.append(aux)  # Select the sites we want to calculate the marker on


for num_rea in range(0, n_realisations):

    x, y, z = GaussianPointSet_3D(x, y, z, width)  # Positions of the sites in the amorphous lattice
    marker = np.zeros((len(list_sites),))  # Declaration of the local marker

    # Iterate in the mass parameter
    for index in range(0, len(M_inset)):

        # Hamiltonian
        H_periodic, matrix_neighbours = AmorphousHamiltonian3D_WT(n_sites, n_orb, n_neighbours, L_x, L_y, L_z, x, y,
                        z, M_inset[index], t1, t2, lamb, "Closed")
        energy_PBC, eigenstates_PBC = eigh(H_periodic)
        idx = energy_PBC.argsort()
        energy_PBC = energy_PBC[idx]  # Ordered energy eigenvalues
        eigenstates_PBC = eigenstates_PBC[:, idx]  # Ordered eigenstates

        # Valence band projector
        U = np.zeros((n_states, n_states), complex)  # Matrix for the projector
        U[:, 0: n_particles] = eigenstates_PBC[:, 0: n_particles]
        P = U @ np.conj(np.transpose(U))  # Projector onto the occupied subspace

        # Local marker calculation (amorphous average for free in just one realisation)
        for site in range(len(list_sites)):
            marker[site] = local_marker_DIII(n_orb, L_x, L_y, L_z, x, y, z, P, S, list_sites[site])

        marker_inset[index, :] = marker_inset[index, :] + marker  # Cumulative marker

    marker_inset = marker_inset / n_realisations  # Average of the marker over different realisations



# %% Plot
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Main figure
fig, ax = plt.subplots(figsize=(8, 6))
axcolour = ['#365365', '#71250E', '#6F6534']
for index in range(0, n_sizes):
    ax.plot(M, marker_final[index, :], color=axcolour[index], marker='.', markersize=12, label='$L= $' + str(size[index]))  # Plot of the gap PBC
    # plt.plot(M, gap_average_PBC[index, :], '.b')  # Plot of the gap PBC

ax.set_ylabel("$cs$", fontsize=20)
ax.set_xlabel("$M$", fontsize=20)
ax.set_xlim(-4, 4)
ax.set_ylim(-2.2, 1.2)

ax.tick_params(which='major', width=0.75)
ax.tick_params(which='major', length=14)
ax.tick_params(which='minor', width=0.75)
ax.tick_params(which='minor', length=7)

majorsy = [-2, -1, 0, 1]
minorsy = [-2.2, -1.5, -0.5, 0.5, 1.2]

ax.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
ax.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))

majorsx = [-4,  -2,  0,  2, 4]
minorsx = [-3, -1, 1, 3]

ax.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
ax.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))

ax.legend(loc='upper right', frameon=False)
ax.text(-3.5, -1.75, " $w=$" + str(width), fontsize=20)
ax.text(-3.5, -2, " $N_{\\rm{samples}}=$" + str(n_realisations), fontsize=20)

plt.tight_layout()
# plt.title(" $N_{\\rm{samples}}=$" + str(n_realisations) + " $w=$" + str(width),fontsize=20)



# Inset figure
left, bottom, width, height = [0.76, 0.3, 0.2, 0.2]
inset_ax = fig.add_axes([left, bottom, width, height])
insetcolour = ['#BF7F04', '#BF5B05', '#8C1C04']

for index in range(len(M_inset)):
    inset_ax.plot(range(0, len(list_sites)), marker_inset[index, :], color=insetcolour[index], marker='.', markersize=8, label='$M=$' + str(M_inset[index]))

inset_ax.set_ylabel("$cs$", fontsize=20)
inset_ax.set_xlabel("$x$", fontsize=20)
inset_ax.set_xlim(0, 11)
inset_ax.set_ylim(-2.5, 2)

inset_ax.tick_params(which='major', width=0.75)
inset_ax.tick_params(which='major', length=14)
inset_ax.tick_params(which='minor', width=0.75)

inset_ax.tick_params(which='minor', length=7)
majorsy2 = [-2, -1, 0, 1, 2]
# minorsy2 = [-2.2, -1.5, -0.5, 0.5, 1.2]

inset_ax.yaxis.set_major_locator(ticker.FixedLocator(majorsy2))
# inset_ax.yaxis.set_minor_locator(ticker.FixedLocator(minorsy2))

majorsx2 = [0, 5.5, 11]
minorsx2 = [2.75, 8.25]

inset_ax.xaxis.set_major_locator(ticker.FixedLocator(majorsx2))
inset_ax.xaxis.set_minor_locator(ticker.FixedLocator(minorsx2))

inset_ax.legend(loc=(0, 0.9), mode="expand", ncol=3, frameon=False)
plt.show()



end_time = time.time()
print("time elapsed=" + str(end_time - start_time) + "s")