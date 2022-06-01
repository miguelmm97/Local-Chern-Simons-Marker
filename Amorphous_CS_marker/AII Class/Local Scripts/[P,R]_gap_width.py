# AII 3D Amorphous model: Seeing if i[P, R] is gapped for different parameters

import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from numpy import pi
from functions import GaussianPointSet_3D, AmorphousHamiltonian3D_WT, spectrum, displacement


start_time = time.time()
# %%  Global definitions

# Parameters of the model
n_orb = 4                          # Number of orbitals per site
n_neighbours = 6                   # Number of neighbours
width = np.linspace(0, 0.2, 100)   # Width of the gaussian for the WT model
density = 1                        # Point density of the RPS model
M = 2                              # Vector of mass parameters in units of t1
t1, t2, lamb = 1, 0.5, 1           # Hopping and spin-orbit coupling in WT model
mu = 0                             # Disorder strength
size = 10

# Pauli matrices
sigma_0 = np.eye(2)                       # Pauli 0
sigma_x = np.array([[0, 1], [1, 0]])      # Pauli x
sigma_y = np.array([[0, -1j], [1j, 0]])   # Pauli y
sigma_z = np.array([[1, 0], [0, -1]])     # Pauli Z

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

# Declarations
gap_PR = np.zeros((len(width),))
gap_H = np.zeros((len(width),))
loc_length = np.zeros([len(width), ])          # Localisation length
Q1r = np.zeros([len(width), ], complex)        # Declaration of Q(r-r')


#%%  Main

# Lattice, trivial Projector and chiral symmetry
lattice(size, n_orb)                         # Crystalline lattice structure
R = np.zeros((n_states, n_states), complex)  # Matrix for the trivial projector
S = -np.kron(sigma_0, sigma_y)               # Chiral symmetry operator for the DIII class
for index in range(0, n_sites):
    R[index * n_orb: index * n_orb + n_orb, index * n_orb: index * n_orb + n_orb] = 0.5 * (np.eye(4) - S)


# Hamiltonian for the AII class
for i, w in enumerate(width):
    print(str(i) + "/" + str(len(width)))

    # Spectrum of H
    x_am, y_am, z_am = GaussianPointSet_3D(x, y, z, w)  # Positions of the sites in the amorphous lattice
    H_periodic, _ = AmorphousHamiltonian3D_WT(n_sites, n_orb, n_neighbours, L_x, L_y, L_z, x_am, y_am, z_am, M, t1, t2, lamb, "Closed")
    energy_PBC, eigenstates_PBC, P = spectrum(H_periodic, n_particles)

    # Spectrum of i[P, R]
    H_aux = 1j * (P @ R - R @ P)  # Auxiliary operator
    vals, vecs = eigh(H_aux)      # Spectrum
    idx = vals.argsort()          # Ordered indexes
    vals = vals[idx]              # Ordered energy eigenvalues
    vecs = vecs[:, idx]           # Ordered eigenstates
    Q = 0.5 * vecs @ (np.diag(np.ones((n_states,)) - np.sign(vals))) @ np.conj(vecs.T)  # ( eigenvalues 0 and 1 projector)

    # Localisation of Q(r-r')
    for j in range(n_sites):
        psi_1 = np.zeros([n_states, ], complex)                          # State  |j, a, up>
        psi_1[j * n_orb + 0], x1, y1, z1 = 1, x_am[j], y_am[j], z_am[j]  # State  |j, a, up> and position values

        # Comparison to other sites
        for k in range(n_sites):
            r_vec, _, _ = displacement(x1, y1, z1, x_am[k], y_am[k], z_am[k], L_x, L_y, L_z, "Closed")
            if r_vec > (np.sqrt(3) * (size / 2) - 1):
                psi_r = np.zeros([n_states, ], complex)                          # State  |k, a, up>
                psi_r[k * n_orb + 0], x2, y2, z2 = 1, x_am[k], y_am[k], z_am[k]  # State  |k, a, up> and position values
                if np.abs(np.abs(np.conj(psi_1.T) @ Q @ psi_r)) > Q1r[i]:
                    Q1r[i] = np.abs(np.abs(np.conj(psi_1.T) @ Q @ psi_r))        # < j | Q | k >



    gap_H[i] = energy_PBC[int(n_states / 2)] - energy_PBC[int(n_states / 2) - 1]  # Gap spectrum
    gap_PR[i] = vals[int(n_states / 2)] - vals[int(n_states / 2) - 1]             # Gap i[P, R]
    loc_length[i] = 100 * Q1r[i]                                                  # Scaled localisation parameter



# Plots
plt.plot(width, gap_H, '.b', markersize=6)       # Plot of the energy
plt.plot(width, gap_PR, '.r', markersize=6)      # Plot of the energy
plt.plot(width, loc_length, '.m', markersize=6)  # Plot of the energy
plt.ylim(0, 2)
plt.xlim(0, width[-1])
plt.xlabel("width")
plt.ylabel("gaps")
plt.title("$L=$" + str(size) + " $M=$" + str(M) + " $t2=$" + str(t2) + " $S= -\sigma_0 \otimes \sigma_y$")
plt.legend(("$\Delta E$", "$\Delta i[P,R]$", "$100 Q(r-r')_{max}$"))
plt.show()


end_time = time.time()
print("time elapsed=" + str(end_time - start_time) + "s")