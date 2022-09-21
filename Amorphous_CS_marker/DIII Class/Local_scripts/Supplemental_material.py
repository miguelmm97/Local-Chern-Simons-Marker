
import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.linalg import eigh
import matplotlib.ticker as ticker
from numpy import pi
from functions import GaussianPointSet_3D, AmorphousHamiltonian3D_WT, local_marker_DIII_OBC, spectrum
from random import sample



start_time = time.time()
# %%  Global definitions

# Parameters of the model
n_orb = 4                       # Number of orbitals per site
n_neighbours = 6                # Number of neighbours
width = 0.1                     # Width of the gaussian for the WT model
density = 1                     # Point density of the RPS model
M = [0]                         # Vector of mass parameters in units of t1
t1, t2, lamb = 1, 0, 1          # Hopping and spin-orbit coupling in WT model
mu = 0                          # Disorder strength
size = 12                      # System size

# Symmetries
sigma_0 = np.eye(2)                      # Pauli 0
sigma_y = np.array([[0, -1j], [1j, 0]])  # Pauli y
s = - np.kron(sigma_0, sigma_y)          # Chiral symmetry operator for the WT model

# Global variables
L_x, L_y, L_z = None, None, None
n_sites, n_states, n_particles = None, None, None
sites, x, y, z = None, None, None, None
S, T = None, None  # Chiral symmetry and TRS

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

# %% Calculation of the marker as a function of the sites (Inset figure)
# Underlying crystalline lattice
lattice(size, n_orb, chiral_symmetry=s)
list_sites, list_x, list_y, list_z, av_marker = [], [], [], [], []

# Amorphous lattice and markers
x, y, z = GaussianPointSet_3D(x, y, z, width)  # Positions of the sites in the amorphous lattice
for j, site in enumerate(sites):
    if L_y/2 - 1.5 < y[j] < L_y/2 + 1.5 and L_z/2 - 1.5 < z[j] < L_z/2 + 1.5:
        list_sites.append(site)
        list_x.append(x[j])
        list_y.append(y[j])
        list_z.append(z[j])
marker_final = np.zeros((len(M), len(list_sites)))    # Declaration of the marker average
marker = np.zeros((len(list_sites),))          # Declaration of the local marker

# Iterate in the mass parameter
for index in range(0, len(M)):

    # Hamiltonian
    H_OBC, matrix_neighbours = AmorphousHamiltonian3D_WT(n_sites, n_orb, n_neighbours,
                                            L_x, L_y, L_z, x, y, z, M[index], t1, t2, lamb, "Open")
    energy_OBC, eigenstates_OBC, P = spectrum(H_OBC, n_particles)  # Eigenstates and valence band projector

    # Local marker calculation (amorphous average for free in just one realisation)
    for j, site in enumerate(list_sites):
        print(str(j) + "/" + str(len(list_sites)))
        marker[j] = local_marker_DIII_OBC(n_orb, x, y, z, P, S, site)
    marker_final[index, :] = marker_final[index, :] + marker  # Cumulative marker

# Average at each crystalline point
for j in range(size):
    average, count = 0, 0
    for k, nu in enumerate(marker_final[0, :]):
        if j - 0.5 < list_x[k] < j + 0.5:
            average = average + nu
            count = count + 1
    average = average / count
    av_marker.append(average)


# %% Main figure
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 13, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=20)
fig, ax = plt.subplots(figsize=(8, 6))
axcolour = ['#FF416D', '#3F6CFF', '#00B5A1'] # light
ax.plot(list_x, marker_final[0, :], '.', color=axcolour[2], markersize=12, label='$L= 8$')
ax.plot(range(size), av_marker, color=axcolour[1], markersize=12, label='$L= 8$')


# Axis labels and limits
ax.set_ylabel("$\\nu$", fontsize=25)
ax.set_xlabel("$x$", fontsize=25)
ax.set_xlim(-1, 13)
ax.set_ylim(-3, 2)
# ax.yaxis.set_label_coords(-0.13, 0.5)

# Axis ticks
ax.tick_params(which='major', width=0.75)
ax.tick_params(which='major', length=14)
ax.tick_params(which='minor', width=0.75)
ax.tick_params(which='minor', length=7)
majorsy = [-3, -2, -1, 0, 1, 2]
minorsy = [-2.5, -1.5, -0.5, 0.5, 1.5]
majorsx = [0, 2, 4, 6, 8, 10, 12]
minorsx = [1, 3, 5, 7, 9, 11]
ax.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
ax.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
ax.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
ax.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))

plt.show()
