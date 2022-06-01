# DIII 3D Amorphous model: Gap/Marker as a function of the mass parameter

import numpy as np
import h5py
import argparse
from functions import GaussianPointSet_3D, AmorphousHamiltonian3D_WT, local_marker_DIII, spectrum

# Arguments to submit to the cluster
parser = argparse.ArgumentParser(description='Argument parser for DIII')
parser.add_argument('-l', '--line', type=int, help='Select line number', default=None)
parser.add_argument('-f', '--file', type=str, help='Select file name', default='params_M_inset.txt')
parser.add_argument('-M', '--outdir', type=str, help='Select the base name of the output file', default='outdir_M')
parser.add_argument('-o', '--outbase', type=str, help='Select the base name of the output file', default='out_M_inset')
args = parser.parse_args()

# Variables that we iterate with the cluster
R = None  # Number of realisation
M = None  # Mass parameter

# Input data
if args.line is not None:
    print("Line number:", args.line)
    with open(args.file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == args.line:
                params = line.split()
                M = float(params[0])
                R = int(params[1])
else:
    raise IOError("No line number was given")

# %%  Global definitions

# Parameters of the model
n_orb = 4               # Number of orbitals per site
n_neighbours = 6        # Number of neighbours
width = 0.1             # Width of the gaussian for the WT model
density = 1             # Point density of the RPS model
t1, t2, lamb = 1, 0, 1  # Hopping and spin-orbit coupling in WT model
mu = 0                  # Disorder strength
size = 12

# Global variables
L_x, L_y, L_z = None, None, None
n_sites, n_states, n_particles = None, None, None
sites, x, y, z = None, None, None, None
S, T = None, None

# Symmetries
sigma_0 = np.eye(2)                      # Pauli 0
sigma_y = np.array([[0, -1j], [1j, 0]])  # Pauli y
s = - np.kron(sigma_0, sigma_y)          # Chiral symmetry operator for the WT model

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
# Calculation of the marker as a function of the sites (Inset figure)
# Underlying crystalline lattice
lattice(size, n_orb, chiral_symmetry=s)
x, y, z = GaussianPointSet_3D(x, y, z, width)  # Positions of the sites in the amorphous lattice
list_sites = []
for index in range(L_x):
    aux = int(0.5 * (L_x * L_y * L_z) + 0.5 * (L_x * L_y) + index)  # Central array of sites
    list_sites.append(aux)
marker = np.zeros((len(list_sites),))  # Declaration of the local marker


# Hamiltonian
H_periodic, matrix_neighbours = AmorphousHamiltonian3D_WT(n_sites, n_orb, n_neighbours,
                                                L_x, L_y, L_z, x, y, z, M, t1, t2, lamb, "Closed")
energy_PBC, eigenstates_PBC, P = spectrum(H_periodic, n_particles)  # Eigenstates and valence band projector

# Local marker calculation (amorphous average for free in just one realisation)
for site in range(len(list_sites)):
    marker[site] = local_marker_DIII(n_orb, L_x, L_y, L_z, x, y, z, P, S, list_sites[site])



# Output data
outfile = '{}-{}'.format(args.outbase, args.line)
with h5py.File(outfile + '.h5', 'w') as f:
    f.create_dataset("data", data=marker)
    f["data"].attrs.create("size", data="inset")
    f["data"].attrs.create("M", data=M)
    f["data"].attrs.create("R", data=R)
    f["data"].attrs.create("width", data=width)


# # Output data
# outfile = '{}-{}.txt'.format(args.outbase, args.line)
# with open(outfile, 'w') as f:
#     f.write('{:.16f}\n'.format(marker_average))
