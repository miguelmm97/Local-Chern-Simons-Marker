# AII 3D Amorphous model: Marker as a function of t2 for fixed M

import numpy as np
import time
from numpy.linalg import eigh
from functions import GaussianPointSet_3D, AmorphousHamiltonian3D_WT, local_marker_AII, spectrum
from random import sample
import h5py
import argparse
import os
from numpy.linalg import eigh

# Arguments to submit to the cluster
parser = argparse.ArgumentParser(description='Argument parser for DIII')
parser.add_argument('-l', '--line', type=int, help='Select line number', default=None)
parser.add_argument('-f', '--file', type=str, help='Select file name', default='params_width.txt')
parser.add_argument('-M', '--outdir', type=str, help='Select the base name of the output file', default='outdir_width')
parser.add_argument('-o', '--outbase', type=str, help='Select the base name of the output file', default='out_width')
args = parser.parse_args()

# Variables that we iterate with the cluster
size = None          # System size
realisations = None  # Number of realisation
M = None             # Mass parameter
t2 = None            # Chiral symmetry breaking
width = None         # Width

# Input data
if args.line is not None:
    print("Line number:", args.line)
    with open(args.file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == args.line:
                params = line.split()
                size = int(params[0])
                M = int(params[1])
                width = float(params[2])
                t2 = float(params[3])
                realisations = int(params[4])
                print(params)
else:
    raise IOError("No line number was given")

if size is None:
    raise ValueError("Size is none")
if M is None:
    raise ValueError("M is none")
if t2 is None:
    raise ValueError("t2 is none")
if width is None:
    raise ValueError("Width is none")
if realisations is None:
    raise ValueError("realisations is none")
# %%  Global definitions

# Parameters of the model
n_orb = 4                         # Number of orbitals per site
n_neighbours = 6                  # Number of neighbours
density = 1                       # Point density of the RPS model
t1, lamb = 1, 1                   # Hopping and spin-orbit coupling in WT model
mu = 0                            # Disorder strength

# Pauli matrices
sigma_0 = np.eye(2)                      # Pauli 0
sigma_x = np.array([[0, 1], [1, 0]])     # Pauli x
sigma_y = np.array([[0, -1j], [1j, 0]])  # Pauli y
sigma_z = np.array([[1, 0], [0, -1]])    # Pauli Z
tau_0, tau_x, tau_y, tau_z = sigma_0, sigma_x, sigma_y, sigma_z
s = -np.kron(sigma_0, tau_y)             # Chiral symmetry operator for the DIII class (spin, orbital)

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


# %% Main


# Amorphous lattice and markers
lattice(size, n_orb)                                    # Crystalline lattice structure
x_am, y_am, z_am = GaussianPointSet_3D(x, y, z, width)  # Positions of the sites in the amorphous lattice
sample_sites = 25                                        # Number of sites we want to take the average over
list_sites = sample(range(0, n_sites), sample_sites)    # Select the sites we want to calculate the marker on
marker = np.zeros((sample_sites, ))

# Trivial Projector
R = np.zeros((n_states, n_states), complex)  # Matrix for the trivial projector
for index in range(0, n_sites):
    R[index * n_orb: index * n_orb + n_orb, index * n_orb: index * n_orb + n_orb] = 0.5 * (np.eye(4) - s)

# Hamiltonian
H_periodic, _ = AmorphousHamiltonian3D_WT(n_sites, n_orb, n_neighbours, L_x, L_y, L_z, x_am, y_am, z_am, M, t1, t2, lamb, "Closed")
energy_PBC, eigenstates_PBC, P = spectrum(H_periodic, n_particles)  # Eigenstates and valence band projector

# DIII Projector
Q_aux1 = 1j * (P @ R - R @ P)  # Auxiliary operator
vals, vecs = eigh(Q_aux1)      # (real eigenvalues)
Q = 0.5 * vecs @ (np.diag(np.ones((n_states,)) - np.sign(vals))) @ np.conj(vecs.T)  # ( eigenvalues 0 and 1 projector)

# Local marker calculation
for n, site in enumerate(list_sites):
    marker[n] = local_marker_AII(n_orb, L_x, L_y, L_z, x_am, y_am, z_am, P, Q, R, site)
marker_average = np.mean(marker)  # Average marker over sites

print("done")

# Output data
outfile1 = '{}-{}'.format(args.outbase, args.line)
outfile2 = os.path.join(args.outdir, outfile1)
with h5py.File(outfile2 + '.h5', 'w') as f:
    f.create_dataset("data", data=marker_average)
    f["data"].attrs.create("size", data=size)
    f["data"].attrs.create("M", data=M)
    f["data"].attrs.create("t2", data=t2)
    f["data"].attrs.create("width", data=width)
    f["data"].attrs.create("R", data=realisations)


print("done")
