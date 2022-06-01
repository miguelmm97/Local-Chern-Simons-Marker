# AII 3D Amorphous model: Seeing if i[P, R] is gapped
import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from numpy import pi
from functions import GaussianPointSet_3D, AmorphousHamiltonian3D_WT, spectrum


start_time = time.time()
# %%  Global definitions

# Parameters of the model
n_orb = 4                    # Number of orbitals per site
n_neighbours = 6             # Number of neighbours
width = 0.1                  # Width of the gaussian for the WT model
density = 1                  # Point density of the RPS model
M = 2                        # Vector of mass parameters in units of t1
t1, t2, lamb = 1, 2, 1       # Hopping and spin-orbit coupling in WT model
mu = 0                       # Disorder strength
size = 5

# Pauli matrices
sigma_0 = np.eye(2)                         # Pauli 0
sigma_x = np.array([[0, 1], [1, 0]])        # Pauli x
sigma_y = np.array([[0, -1j], [1j, 0]])     # Pauli y
sigma_z = np.array([[1, 0], [0, -1]])       # Pauli Z

# Global variables
L_x, L_y, L_z = None, None, None
n_sites, n_states, n_particles = None, None, None
sites, x, y, z = None, None, None, None

# Lattice definition
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
lattice(size, n_orb)
x, y, z = GaussianPointSet_3D(x, y, z, width)  # Positions of the sites in the amorphous lattice


#%%  Main

# Hamiltonian for the AII class
H_periodic, _ = AmorphousHamiltonian3D_WT(n_sites, n_orb, n_neighbours, L_x, L_y, L_z, x, y, z, M, t1, t2, lamb, "Closed")
energy_PBC, eigenstates_PBC, P = spectrum(H_periodic, n_particles)

# Trivial Projector
R = np.zeros((n_states, n_states), complex)  # Matrix for the trivial projector
S = -np.kron(sigma_0, sigma_y)                # Chiral symmetry operator for the DIII class (in orbital basis only)
for index in range(0, n_sites):
    R[index * n_orb: index * n_orb + n_orb, index * n_orb: index * n_orb + n_orb] = 0.5 * (np.eye(4) - S)


# Spectrum of i[P, R]
H_aux = 1j * (P @ R - R @ P)  # Auxiliary operator
val, vec = eigh(H_aux)        # Spectrum
idx = val.argsort()           # Ordered indexes
val = val[idx]                # Ordered energy eigenvalues
vec = vec[:, idx]             # Ordered eigenstates


# Plots
plt.plot(range(0, n_states), val, '.b', markersize=6)  # Plot of the energy
# plt.ylim(-0.1, 0.1)
plt.xlim(0, n_states)
plt.xlabel("Eigenstate number")
plt.ylabel("Eigenvalue")
plt.title("$i[P,R]$, $S= \sigma_x$")
plt.show()


plt.plot(range(0, n_states), energy_PBC, '.r', markersize=6)  # Plot of the energy
# plt.ylim(-0.1, 0.1)
plt.xlim(0, n_states)
plt.xlabel("Eigenstate number")
plt.ylabel("Eigenvalue")
plt.show()

end_time = time.time()
print("time elapsed=" + str(end_time - start_time) + "s")