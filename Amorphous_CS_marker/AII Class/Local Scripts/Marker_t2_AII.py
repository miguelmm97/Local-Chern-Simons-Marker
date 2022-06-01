# AII 3D Amorphous model: Marker as a function of t2 for fixed M

import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.linalg import eigh
import matplotlib.ticker as ticker
from numpy import pi
from functions import GaussianPointSet_3D, AmorphousHamiltonian3D_WT, local_marker_AII, spectrum
from random import sample



start_time = time.time()
# %%  Global definitions

# Parameters of the model
n_orb = 4                         # Number of orbitals per site
n_neighbours = 6                  # Number of neighbours
width = [0.025, 0.05]   # Width of the gaussian for the WT model
density = 1                       # Point density of the RPS model
M = 0                             # Mass parameters in units of t1
t1, lamb = 1, 1                   # Hopping and spin-orbit coupling in WT model
t2 = np.linspace(0, 2, 50)        # Chiral symmetry breaking
mu = 0                            # Disorder strength
sizes = [6]                       # System size


# Pauli matrices
sigma_0 = np.eye(2)                      # Pauli 0
sigma_x = np.array([[0, 1], [1, 0]])     # Pauli x
sigma_y = np.array([[0, -1j], [1j, 0]])  # Pauli y
sigma_z = np.array([[1, 0], [0, -1]])    # Pauli Z
tau_0, tau_x, tau_y, tau_z = sigma_0, sigma_x, sigma_y, sigma_z
s = np.kron(sigma_z, tau_x)             # Chiral symmetry operator for the DIII class (spin, orbital)


# Global variables
L_x, L_y, L_z = None, None, None
n_sites, n_states, n_particles = None, None, None
sites, x, y, z = None, None, None, None
S, T = None, None  # Chiral symmetry and TRS

# Average over different realisations
n_realisations = 1                                           # Realisations for each system size
n_sizes = len(sizes)                                         # Different system sizes for each block of realisations
marker_final = np.zeros((n_sizes, len(width), len(t2)))      # Declaration of the marker average

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

# Iterate number of sites
for i, size in enumerate(sizes):

    # Underlying crystalline lattice
    lattice(size, n_orb)                                  # Crystalline lattice structure
    sample_sites = 1                                      # Number of sites we want to take the average over
    list_sites = sample(range(0, n_sites), sample_sites)  # Select the sites we want to calculate the marker on

    # Trivial Projector
    R = np.zeros((n_states, n_states), complex)  # Matrix for the trivial projector
    for index in range(0, n_sites):
        R[index * n_orb: index * n_orb + n_orb, index * n_orb: index * n_orb + n_orb] = 0.5 * (np.eye(4) - s)

    # Different widths
    for k, w in enumerate(width):

        # Iterate number of realisations
        for sample in range(n_realisations):

            # Amorphous lattice and markers
            x_am, y_am, z_am = GaussianPointSet_3D(x, y, z, w)   # Positions of the sites in the amorphous lattice
            marker = np.empty((sample_sites,))          # Declaration of the local marker of each site
            marker_average = np.empty((len(t2), ))      # Declaration of the marker average over sites
            marker_error = np.empty((len(t2),))         # Declaration of the marker error over sites

            # Iterate in t2
            for j, t in enumerate(t2):
                print("size=" + str(i) + ", realisation=" + str(sample) + ", t2=" + str(j))  # Sanity check

                # Hamiltonian
                H_periodic, _ = AmorphousHamiltonian3D_WT(n_sites, n_orb, n_neighbours, L_x, L_y, L_z, x_am, y_am, z_am, M, t1, t, lamb, "Closed")
                energy_PBC, eigenstates_PBC, P = spectrum(H_periodic, n_particles)  # Eigenstates and valence band projector

                # DIII Projector
                Q_aux1 = 1j * (P @ R - R @ P)  # Auxiliary operator
                vals, vecs = eigh(Q_aux1)      # (real eigenvalues)
                Q = 0.5 * vecs @ (np.diag(np.ones((n_states,)) - np.sign(vals))) @ np.conj(vecs.T)  # ( eigenvalues 0 and 1 projector)


                # Local marker calculation
                for n, site in enumerate(list_sites):
                    marker[n] = local_marker_AII(n_orb, L_x, L_y, L_z, x_am, y_am, z_am, P, Q, R, site)
                marker_average[j] = np.mean(marker)                         # Average marker over sites
                marker_error[j] = np.std(marker) / np.sqrt(sample_sites)    # Standard error mover sites

            marker_final[i, k, :] = marker_final[i, k, :] + marker_average  # Cumulative marker adding each realisation

        marker_final[i, k, :] = marker_final[i, k, :] / n_realisations      # Average marker per realisation



# %% Plots

# Font format
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Main figure
fig, ax = plt.subplots(figsize=(8, 6))
axcolour = ['#365365', '#71250E', '#6F6534', '#9932CC']
for i in range(n_sizes):
    for j in range(len(width)):
        ax.plot(t2, marker_final[i, j, :], color=axcolour[j], marker='.', markersize=12, label='$L= $'+ str(sizes[i]) + " $w=$" + str(width[j]))
ax.plot(t2, np.repeat(0, len(t2)), '--')
ax.plot(t2, np.repeat(2, len(t2)), '--')

# Axis labels and limits
ax.set_ylabel("$\\nu$", fontsize=20)
ax.set_xlabel("$t_2/t_1$", fontsize=20)
ax.set_xlim(0, 2)
ax.set_ylim(-0.1, 2.25)

# Axis ticks
#ax.tick_params(which='major', width=0.75)
#ax.tick_params(which='major', length=14)
ax.tick_params(which='minor', width=0.75)
ax.tick_params(which='minor', length=7)
#majorsy = [0, 1, 2]
#minorsy = [-0.2, 0.5, 1, 1.5, 2.2]
majorsx = [0, 0.5, 1, 1.5, 2]
minorsx = [0.25, 0.75, 1.25, 1.75]
#ax.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
#ax.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
ax.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
ax.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))

# Legend and inset text
ax.legend(loc='upper right', frameon=False, fontsize=20)
ax.text(1.5, 0.5, "$L=$" + str(size) + " $M=$" + str(M), fontsize=20)
ax.text(1.5, 0.4," $S= -\sigma_0 \otimes \sigma_y$", fontsize=20)
# ax.text(-4.5, -1.5, " $N=$" + str(n_realisations), fontsize=20)

plt.tight_layout()
# plt.title(" $N_{\\rm{samples}}=$" + str(n_realisations) + " $w=$" + str(width),fontsize=20)


plt.show()



end_time = time.time()
print("time elapsed=" + str(end_time - start_time) + "s")