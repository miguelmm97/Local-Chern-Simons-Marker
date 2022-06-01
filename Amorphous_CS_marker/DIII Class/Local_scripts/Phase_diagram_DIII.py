# Phase diagram in M-w for the DIII class

import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from numpy import pi
from random import sample
import argparse
import h5py
from functions import GaussianPointSet_3D, AmorphousHamiltonian3D_WT, local_marker_DIII, spectrum

# # Arguments to submit to the cluster
parser = argparse.ArgumentParser(description='Argument parser for DIII')
parser.add_argument('-l', '--line', type=int, help='Select line number', default=None)
parser.add_argument('-f', '--file', type=str, help='Select file name', default='params_pd.txt')
parser.add_argument('-M', '--outdir', type=str, help='Select the base name of the output file', default='outdir_pd')
parser.add_argument('-o', '--outbase', type=str, help='Select the base name of the output file', default='out_pd')
args = parser.parse_args()

# Variables that we iterate with the cluster
realisations = None  # Number of realisation
M = None             # Mass parameter
width = None         # Width

# Input data
if args.line is not None:
    print("Line number:", args.line)
    with open(args.file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == args.line:
                params = line.split()
                M = float(params[0])
                width = float(params[1])
                realisations = int(params[2])
                print(params)
else:
    raise IOError("No line number was given")

# %%  Global definitions

# Parameters of the model
n_orb = 4               # Number of orbitals per site
n_neighbours = 6        # Number of neighbours
density = 1             # Point density of the RPS model
t1, t2, lamb = 1, 0, 1  # Hopping and spin-orbit coupling in WT model
mu = 0                  # Disorder strength
size = 8                # System size

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

# Calculation of the marker with M (main figure)

# Underlying crystalline lattice
lattice(size, n_orb, chiral_symmetry=s)               # Crystalline lattice definition, symmetries, ... etc
x, y, z = GaussianPointSet_3D(x, y, z, width)         # Positions of the sites in the amorphous lattice
sample_sites = 20                                     # Number of sites we want to take the average over
list_sites = sample(range(n_sites), sample_sites)  # Select the sites we want to calculate the marker on
marker = np.zeros((sample_sites,))                    # Declaration of the local marker

# Hamiltonian
H_periodic, _ = AmorphousHamiltonian3D_WT(n_sites, n_orb, n_neighbours, L_x, L_y, L_z, x, y, z, M, t1, t2, lamb, "Closed")
energy_PBC, eigenstates_PBC, P = spectrum(H_periodic, n_particles)  # Ordered eigenstates and valence band projector
gap_PBC = energy_PBC[int(n_states / 2)] - energy_PBC[int(n_states / 2) - 1]  # Energy gap


# Local marker calculation
for i, site in enumerate(list_sites):
    print(site)
    marker[i] = local_marker_DIII(n_orb, L_x, L_y, L_z, x, y, z, P, S, site)
marker_average = np.mean(marker)  # Average of the marker for the specified M over sites

exit(0)
# Output data
outfile1 = '{}-{}'.format(args.outbase, args.line)
outfile2 = os.path.join(args.outdir, outfile1)
with h5py.File(outfile2 + '.h5', 'w') as f:
    f.create_dataset("data", data=[marker_average, gap_PBC])
    f["data"].attrs.create("M", data=M)
    f["data"].attrs.create("width", data=width)
    f["data"].attrs.create("R", data=realisations)

print("done")