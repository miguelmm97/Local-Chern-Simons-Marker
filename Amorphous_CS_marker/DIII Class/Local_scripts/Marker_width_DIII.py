# DIII 3D Amorphous model: Gap/Marker as a function of the width

import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from numpy import pi
import matplotlib.ticker as ticker
from functions import GaussianPointSet_3D, AmorphousHamiltonian3D_WT, local_marker_DIII, spectrum

start_time = time.time()
# %%  Global definitions

# Parameters of the model
n_orb = 4                        # Number of orbitals per site
n_neighbours = 6                 # Number of neighbours
width = np.linspace(0, 0.2, 1)  # Width of the gaussian for the WT model
density = 1                      # Point density of the RPS model
M = 0                            # Mass parameter in units of t1
t1, t2, lamb = 1, 0, 1           # Hopping and spin-orbit coupling in WT model
mu = 0                           # Disorder strength
size = [8]                      # System size

# Chiral symmetry
sigma_0 = np.eye(2)                      # Pauli 0
sigma_y = np.array([[0, -1j], [1j, 0]])  # Pauli y
s = - np.kron(sigma_0, sigma_y)          # Chiral symmetry operator for the WT model

# Global variables
L_x, L_y, L_z = None, None, None
n_sites, n_states, n_particles = None, None, None
sites, x, y, z = None, None, None, None
S, T = None, None  # Chiral symmetry and TRS

# Average over different realisations
n_sizes = len(size)
n_widths = len(width)
n_realisations = 1                            # Realisations for each system size
gap_PBC = np.zeros([n_sizes, n_widths])       # Declaration of the vector containing the gaps PBC
marker_final = np.zeros([n_sizes, n_widths])  # Declaration of the local marker average
marker_error = np.zeros([n_sizes, n_widths])  # Declaration of the local marker average
tol = 0.01                                    # Allowed error in the marker
string_legend = []                            # Strings for the legend of the plots
# for index in range(0, n_sizes): string_legend.append(str(size[index]))


# Lattice creation
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
# %% Main

for num_size in range(0, n_sizes):

    # Underlying crystalline lattice
    lattice(size[num_size], n_orb, chiral_symmetry=s)


    # Iterate in the number of widths
    for num_width in range(0, n_widths):

        marker_average = []

        # Iterate in the number of realisations
        for num_rea in range(0, n_realisations):
            print("size=" + str(num_size) + " width=" + str(num_width) + ", realisation=" + str(num_rea))  # Sanity check

            # Amorphous lattice
            x, y, z = GaussianPointSet_3D(x, y, z, width[num_width])  # Positions of the sites in the amorphous lattice

            # Declaration of the local marker
            list_sites = []
            for index in range(L_x):
                aux = int(0.5 * (L_x * L_y * L_z) + 0.5 * (L_x * L_y) + index)  # Central array of sites
                list_sites.append(aux)
            marker = np.zeros((len(list_sites),))  # Declaration of the local marker

            # Hamiltonian
            H_periodic, matrix_neighbours = AmorphousHamiltonian3D_WT(n_sites, n_orb, n_neighbours,
                                                                      L_x, L_y, L_z, x, y, z, M, t1, t2, lamb, "Closed")
            energy_PBC, eigenstates_PBC, P = spectrum(H_periodic, n_particles)  # Eigenstates and valence band projector
            gap_PBC[num_size, num_width] = energy_PBC[int(n_states / 2)] - energy_PBC[int(n_states / 2) - 1]  # Gap


            # Local marker calculation (amorphous average for free in just one realisation)
            for site in range(len(list_sites)):
                marker[site] = local_marker_DIII(n_orb, L_x, L_y, L_z, x, y, z, P, S, list_sites[site])
            marker_average.append(np.mean(marker))
            marker_error[num_size, num_width] = np.std(marker_average)
            if num_rea > 1 and marker_error[num_size, num_width] < tol:
                print("realisations until tolerance: " + str(num_rea))
                break


        gap_PBC[num_size, num_width] = gap_PBC[num_size, num_width] / n_realisations
        marker_final[num_size, num_width] = np.mean(marker_average)
        marker_error[num_size, num_width] = np.std(marker_average)





# %% Plots
exit(0)
# Font format
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Main figure
fig, ax = plt.subplots(figsize=(8, 6))
axcolour = ['#C00425', '#306367']
axmarkers = ['dashed', 'dotted', 'solid']
for index in range(0, n_sizes):
    ax.plot(width, marker_final[index, :], color='k', linestyle=axmarkers[index], linewidth=2, label='$L= $' + str(size[index]))
    ax.plot(width, marker_final[index, :], color=axcolour[0], linestyle=axmarkers[index], linewidth=2)
    ax.plot(width, gap_PBC[index, :], color=axcolour[1], linestyle=axmarkers[index], linewidth=2)
    
# Axis labels and limits

ax.set_ylabel("$cs$", fontsize=20)
ax.set_xlabel("$w$", fontsize=20)
ax.set_xlim(0, 0.2)
ax.set_ylim(-0.1, 2)

# Axis ticks
ax.tick_params(which='major', width=0.75)
ax.tick_params(which='major', length=14)
ax.tick_params(which='minor', width=0.75)
ax.tick_params(which='minor', length=7)
majorsy = [0, 0.5, 1, 1.5, 2]
minorsy = [0.25, 0.75, 1.25, 1.75]
majorsx = [0, 0.1, 0.2]
minorsx = [0.05, 0.15]
ax.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
ax.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
ax.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
ax.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))

ax.arrow(0.025, 0.75, -0.02, 0, length_includes_head=True,
          head_width=0.05, head_length=0.005, color='#C00425')

ax.arrow(0.175, 1.25, 0.02, 0, length_includes_head=True,
          head_width=0.05, head_length=0.005, color='#306367')

# Legend and inset text
ax.legend(loc='upper right', frameon=False, fontsize=20)


right_ax = ax.twinx()
right_ax.set_ylabel("$E_g/t_1$", fontsize=20)
right_ax.set_ylim(-0.1, 2)

# Axis ticks
right_ax.tick_params(which='major', width=0.75)
right_ax.tick_params(which='major', length=14)
right_ax.tick_params(which='minor', width=0.75)
right_ax.tick_params(which='minor', length=7)
majorsy = [0, 0.5, 1, 1.5, 2]
minorsy = [0.25, 0.75, 1.25, 1.75]
majorsx = [0, 0.1, 0.2]
minorsx = [0.05, 0.15]
right_ax.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
right_ax.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
right_ax.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
right_ax.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))



plt.tight_layout()
# plt.title(" $N_{\\rm{samples}}=$" + str(n_realisations) + " $w=$" + str(width),fontsize=20)
plt.show()