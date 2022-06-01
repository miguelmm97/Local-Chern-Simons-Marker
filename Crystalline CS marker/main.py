# 3D DIII Fu and Berg model

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from numpy.linalg import eigh, norm, inv
from functions import Hamiltonian3D_DIII_FB, Hamiltonian3D_DIII

#%% Global variables

# Parameters of the model
t = 1  # Hopping amplitude. Everything is in terms of this so keep it always to 1
lamb = 2  # Another hopping amplitude in units of t
lamb_z = 1.8  # Spin-orbit coupling in units of t
eps = 5  # Intracell orbital mixing in units of t
flux = 0  # flux
m = [1]  # Mass parameter
string_legend = []  # Strings for the legend of the plots
for index in range(0, len(m)): string_legend.append(str(m[index]))

# Lattice basis definition
n_x, n_y, n_z = 11, 11, 11  # Number of sites on each direction
n_orb = 4  # Number of orbitals per site
n_states = n_x * n_y * n_z * n_orb  # Total number of states in the basis
sites = np.arange(0,  n_x * n_y * n_z)  # Array with the number of each site
x = sites % n_x  # x position
y = (sites // n_x) % n_y  # y position
z = sites // (n_x * n_y)  # z position
n_particles = int(n_states/2)  # Number of particles

# Operator definitions for the Z2 invariant
X, Y, Z = np.zeros((n_states, n_states)), np.zeros((n_states, n_states)), np.zeros((n_states, n_states))  # X,Y,Z Operators
S = np.zeros((n_states, n_states), complex)  # Chiral symmetry operator
Z2_local_invariant = np.zeros((len(m), n_x))
sigma_z = np.array([[1, 0], [0, -1]])
sigma_y = np.array([[0, -1j], [1j, 0]])
for index in range(0, n_x * n_y * n_z):
    X[index * n_orb: index * n_orb + n_orb, index * n_orb: index * n_orb + n_orb] = x[index] * np.eye(4)
    Y[index * n_orb: index * n_orb + n_orb, index * n_orb: index * n_orb + n_orb] = y[index] * np.eye(4)
    Z[index * n_orb: index * n_orb + n_orb, index * n_orb: index * n_orb + n_orb] = z[index] * np.eye(4)
    S[index * n_orb: index * n_orb + n_orb, index * n_orb: index * n_orb + n_orb] = np.kron(np.eye(2), -sigma_y)

#%% Hamiltonian and band structure

# PBC
for index_m in range(len(m)):
    print(m[index_m])
    # Spectrum
    H_PBC = Hamiltonian3D_DIII(n_x, n_y, n_z, n_orb, sites, x, y, z, m[index_m], "Closed")  # Hamiltonian
    energy_PBC, eigenstates_PBC = eigh(H_PBC)  # Each column contains one band (points might not be ordered)
    idx_PBC = energy_PBC.argsort()  # [:n] #add [:n] to only pick out the n lowest eigenvalues.
    eigenstates_PBC = eigenstates_PBC[:, idx_PBC]
    energy_PBC = energy_PBC[idx_PBC]  # Ordered energy eigenvalues

    plt.plot(range(0, n_states), energy_PBC, '.b', markersize=5)
    # plt.plot(basis, energy_OBC, '.r')
    plt.xlim(0, n_states)
    plt.xlabel('# state')
    plt.ylabel('E/t')
    plt.title('3D TI band structure')
    plt.show()

    # Valence band projector
    U = np.zeros((n_states, n_states), complex)
    U[:, 0:n_particles] = eigenstates_PBC[:, 0:n_particles]
    P = U @ np.conj(np.transpose(U))

    # Z2 Invariant calculation
    M = (P @ S @ X @ P @ Y @ P @ Z @ P) + (P @ S @ Z @ P @ X @ P @ Y @ P) + (P @ S @ Y @ P @ Z @ P @ X @ P) - \
        (P @ S @ X @ P @ Z @ P @ Y @ P) - (P @ S @ Z @ P @ Y @ P @ X @ P) - (P @ S @ Y @ P @ X @ P @ Z @ P)
    for index in range(n_x):
        aux = int(0.5 * (n_x * n_y * n_z) + 0.5 * (n_x * n_y) + index)
        Z2_local_invariant[index_m, index] = (8 * pi / 3) * np.imag(np.trace(M[aux * n_orb: aux * n_orb + n_orb,
                                                                             aux * n_orb: aux * n_orb + n_orb]))



for index in range(len(m)):
    plt.plot(range(n_x), Z2_local_invariant[index, :], '.', markersize=6)
plt.ylim(-3, 3)
plt.xlabel('$n_x$')
plt.ylabel('$Z$ local marker')
plt.legend(string_legend)
plt.title("$N_x, N_y, N_z=$" + str(n_x))
plt.show()






# # FIGURES
# plt.plot(basis, energy_PBC, '.b', markersize=5)
# # plt.plot(basis, energy_OBC, '.r')
# plt.xlabel('# state')
# plt.ylabel('E/t')
# plt.title('3D TI band structure')
# plt.show()
#
#
# check1 = M
# check2 = np.conj(np.transpose(M))
# check = check1 + check2
# # print(check[0, 3])
# # print(check)
# for index1 in range(n_states):
#     for index2 in range(n_states):
#         if np.imag(check[index1, index2]) > 0.00001:
#             print([index1, index2])
#             print(check[index1, index2])

# for index in range(0, n_particles):  # Projector to the occupy bands
    #     P = P + np.outer((eigenstates_PBC[:, idx_PBC[index]] / norm(eigenstates_PBC[:, idx_PBC[index]])),
    #              np.conj(eigenstates_PBC[:, idx_PBC[index]] / norm(eigenstates_PBC[:, idx_PBC[index]])))