# FUNCTIONS
# Function directory for the stacked chern amorphous nanowire

import numpy as np
from numpy import e, pi
import matplotlib.pyplot as plt
from random import gauss
from mpl_toolkits.mplot3d import Axes3D


# %%  Definition of the RPS, distance, angle and boundary conditions
def RandomPointSet_3D(n_sites, L_x, L_y, L_z):
    
    # Generate the coordinates of a random point set in 2D
    # n_sites: NUmber of points in the set
    # L_x: Length of the area containing the set in x-direction
    # L_y: Length of the area containing the set in y-direction
    # L_y: Length of the area containing the set in z-direction
    
    x = L_x * np.random.rand(n_sites)  # x-position
    y = L_y * np.random.rand(n_sites)  # y-position
    z = L_z * np.random.rand(n_sites)  # z-position
    return x, y, z

def GaussianPointSet_3D(x, y, z, width):
    
    # Generate a gaussian point set with the specified width
    # x, y, z: Positions for the crystalline case
    # width: Specified with for the gaussian distribution
    
    deltax = np.zeros(len(x))  # Deviations from the crystalline case in x direction
    deltay = np.zeros(len(y))  # Deviations from the crystalline case in y direction
    deltaz = np.zeros(len(z))  # Deviations from the crystalline case in z direction
    
    # Generate some Gaussian values centered at zero with given width
    for index in range(len(x)):
        deltax[index] = gauss(0, width)
    for index in range(len(y)):
        deltay[index] = gauss(0, width)
    for index in range(len(z)):
        deltaz[index] = gauss(0, width)

    x = x + deltax
    y = y + deltay
    z = z + deltaz

    return x, y, z

def displacement(x1, y1, z1, x2, y2, z2, L_x, L_y, L_z, boundary):
    
    # Calculates the displacement vector between sites 2 and 1.
    # x1, y1, z1, x2, y2, z2: Coordinates of the sites
    # L_x, L_y, L_z: System sizes
    # boundary: String containing "Open" or "Closed"

    # Definition of the vector between sites 2 and 1 (from st.1 to st.2)
    v = np.zeros((3,))
    if boundary == "Closed":
        v[0] = (x2 - x1) - L_x * np.sign(x2 - x1) * np.heaviside(abs(x2 - x1) - L_x / 2, 0)
        v[1] = (y2 - y1) - L_y * np.sign(y2 - y1) * np.heaviside(abs(y2 - y1) - L_y / 2, 0)
        v[2] = (z2 - z1) - L_z * np.sign(z2 - z1) * np.heaviside(abs(z2 - z1) - L_z / 2, 0)

    elif boundary == "Open":
        v[0] = (x2 - x1)
        v[1] = (y2 - y1)
        v[2] = (z2 - z1)

    # Module of the vector between sites 2 and 1
    r = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)

    # Phi angle of the vector between sites 2 and 1 (angle in the XY plane)
    if v[0] == 0:
        if v[1] > 0:  # Pathological case, separated to not divide by 0
            phi = pi / 2
        else:
            phi = 3 * pi / 2
    else:
        if v[1] > 0:                               # We take arctan2 because we have 4 quadrants
            phi = np.arctan2(v[1], v[0])           # 1st and 2nd quadrants
        else:
            phi = 2 * pi + np.arctan2(v[1], v[0])  # 3rd and 4th quadrants

    # Theta angle of the vector between sites 2 and 1 (angle from z)
    r_plane = np.sqrt(v[0] ** 2 + v[1] ** 2)  # Auxiliary radius for the xy plane

    if r_plane == 0:        # Pathological case, separated to not divide by 0
        if v[2] > 0:        # Hopping in z
            theta = 0
        elif v[2] < 0:      # Hopping in -z
            theta = pi
        else:
            theta = pi / 2  # XY planes
    else:
        theta = pi / 2 - np.arctan(v[2] / r_plane)  # 1st and 2nd quadrants

    return r, phi, theta

# %% Hopping functions and hamiltonian

def xtranslation(x, y, z, n_x, n_y, n_z):
    
    # Translates the vector x one site in direction x
    # x, y, z: Vectors with the position of the lattice sites
    # n_x, n_y, n_z: Dimension s of the lattice grid
    
    transx = ((x + 1) % n_x) + n_x * y + n_x * n_y * z

    return transx

def ytranslation(x, y, z, n_x, n_y, n_z):
    
    # Translates the vector y one site in direction y
    # x, y, z: Vectors with the position of the lattice sites
    # n_x, n_y, n_z: Dimension s of the lattice grid
    
    transy = x + n_x * ((y + 1) % n_y) + n_x * n_y * z

    return transy

def ztranslation(x, y, z, n_x, n_y, n_z):
    
    # Translates the vector z one site in direction z
    # x, y, z: Vectors with the position of the lattice sites
    # n_x, n_y, n_z: Dimension s of the lattice grid
    
    transz = x + n_x * y + n_x * n_y * ((z + 1) % n_z)

    return transz

def spectrum(H, n_particles):

    # Calculates the spectrum and the valence band projector of the given Hamiltonian
    # H: Hamiltonian for the model
    # n_particles: Number of particles we want (needed for the projector)

    energy, eigenstates = np.linalg.eigh(H)                # Diagonalise H
    idx = energy.argsort()                                 # Indexes from lower to higher energy
    energy = energy[idx]                                   # Ordered energy eigenvalues
    eigenstates = eigenstates[:, idx]                      # Ordered eigenstates

    # Valence band projector
    U = np.zeros((len(H), len(H)), complex)                # Matrix for the projector
    U[:, 0: n_particles] = eigenstates[:, 0: n_particles]  # Filling the projector
    P = U @ np.conj(np.transpose(U))                       # Projector onto the occupied subspace

    return energy, eigenstates, P

def RadialHop(r):

    # Calculates the radial hopping strength for two sites at a distance r
    # r : Distance between sites (in units of a)

    C = e                            # Normalisation constant for the radial hopping (t=1 for r=a)
    hop_amplitude = C * np.exp(- r)  # Hopping amplitude

    return hop_amplitude

def AngularHop(theta, phi, t1, t2, lamb):
    
    # Calculates the angular hopping strength for two sites at an angle theta
    # theta:  Relative angle between sites in z direction
    # phi:  Relative angle between sites in XY cross-section
    # t1, t2, lambda: Parameters of the model
    
    # Pauli matrices
    sigma_0 = np.eye(2)
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    sigma_p = 0.5 * (sigma_x + 1j * sigma_y)
    sigma_m = 0.5 * (sigma_x - 1j * sigma_y)

    # Hamiltonian
    hopping = - 0.5 * t1 * np.kron(sigma_0, sigma_z)
    SO_coupling = 0.5 * 1j * lamb * (np.cos(theta) * np.kron(sigma_z, sigma_x) +
                                     np.exp(-1j * phi) * np.sin(theta) * np.kron(sigma_p, sigma_x) +
                                     np.exp(1j * phi) * np.sin(theta) * np.kron(sigma_m, sigma_x))
    SO_coupling_AII = 0.5 * t2 * (np.cos(theta) * np.kron(sigma_z, sigma_y) +
                                  np.exp(-1j * phi) * np.sin(theta) * np.kron(sigma_p, sigma_y) +
                                  np.exp(1j * phi) * np.sin(theta) * np.kron(sigma_m, sigma_y))

    hop_matrix = hopping + SO_coupling + SO_coupling_AII

    return hop_matrix

def Diagonal(M):
    
    # Calculates the diagonal block associated with the onsite energies and orbital mixing
    # M: Mass parameter of the model
    
    sigma_0 = np.eye(2)                            # Pauli 0
    sigma_z = np.array([[1, 0], [0, -1]])          # Pauli z
    diag_matrix = - M * np.kron(sigma_0, sigma_z)  # Diagonal term

    return diag_matrix

def AmorphousHamiltonian3D_WT(n_sites, n_orb, n_neighbours, L_x, L_y, L_z, x, y, z, M, t1, t2, lamb, boundary):
    
    # Generates the Hamiltonian for a 3D insulator with the parameters specified
    # t2 = 0  we are in the DIII class (S = - sigma0 x sigma_y), if t2 is non-zero we are in the AII class
    # n_sites: Number of sites in the RPS
    # n_orb: Number of orbitals in the model
    # n_neighbours: Number of fixed neighbours
    # x: x position of the sites in the RPS
    # y: y position of the sites in the RPS
    # z: z position of the sites in the RPS
    # M, t1, t2, lamb: Parameters of the AII model
    # Boundary: String with values "Open" or "Closed" which selects the boundary we want

    Hamiltonian = np.zeros((n_orb * n_sites, n_orb * n_sites), complex)                   # Declaration of the H
    matrix_neighbours = np.tile(np.arange(n_sites), n_sites).reshape((n_sites, n_sites))  # Declaration matrix of neigh
    matrix_dist = np.zeros((n_sites, n_sites))                                            # Declaration matrix of dists
    matrix_phis = np.zeros((n_sites, n_sites))                                            # Declaration matrix of phis
    matrix_thetas = np.zeros((n_sites, n_sites))                                          # Declaration matrix of thetas

    # Hamiltonian and neighbour structure
    cont = 0
    for index1 in range(0, n_sites):

        # List of neighbours
        for index2 in range(cont, n_sites):

            if index1 == index2:
                row = index1 * n_orb
                Hamiltonian[row: row + n_orb, row: row + n_orb] = Diagonal(M)  # Diagonal terms
            else:
                r, phi, theta = displacement(x[index1], y[index1], z[index1], x[index2], y[index2], z[index2], L_x, L_y,
                                                                                                         L_z, boundary)
                matrix_dist[index1, index2], matrix_phis[index1, index2], matrix_thetas[index1, index2] = r, phi, theta
                matrix_dist[index2, index1], matrix_phis[index2, index1], matrix_thetas[index2, index1] = r, phi + pi, \
                                                                                                          pi - theta

        idx = matrix_dist[index1, :].argsort()                         # Sorting the distances from minimum to maximum
        matrix_neighbours[index1, :] = matrix_neighbours[index1, idx]  # Ordered neighbours
        matrix_dist[index1, :] = matrix_dist[index1, idx]              # Ordered distances
        matrix_phis[index1, :] = matrix_phis[index1, idx]              # Ordered phis
        matrix_thetas[index1, :] = matrix_thetas[index1, idx]          # Ordered thetas

        # Hamiltonian for the nearest neighbour contributions
        for index2 in range(1, n_neighbours + 1):
            row = index1 * n_orb
            column = int(matrix_neighbours[index1, index2]) * n_orb
            r, phi, theta = matrix_dist[index1, index2], matrix_phis[index1, index2], matrix_thetas[index1, index2]
            Hamiltonian[row: row + n_orb, column: column + n_orb] = RadialHop(r) * AngularHop(theta, phi, t1, t2, lamb)
            Hamiltonian[column: column + n_orb, row: row + n_orb] = \
                np.transpose(np.conj(Hamiltonian[row: row + n_orb, column: column + n_orb]))

    return Hamiltonian, matrix_neighbours


def local_marker_DIII(n_orb, L_x, L_y, L_z, x, y, z, P, S, site):
    
    # Calculates the local CS marker for the specified site
    # L_x, L_y, L_z: Number of lattice sites in each direction
    # n_orb : Number of orbitals
    # n_sites: Number of lattice sites
    # x, y, z: Vectors with the position of the lattice sites
    # P : Valence band projector
    # S: Chiral symmetry operator of the model
    # site: Number of the site we calculate the marker on

    # Position operators for the invariant (take the particular site to be as far from the branchcut as possible)
    half_Lx, half_Ly, half_Lz = np.floor(L_x / 2), np.floor(L_y / 2), np.floor(L_z / 2)
    deltaLx = np.heaviside(x[site] - half_Lx, 0) * abs(x[site] - (half_Lx + L_x)) + \
                                                            np.heaviside(half_Lx - x[site], 0) * abs(half_Lx - x[site])
    deltaLy = np.heaviside(y[site] - half_Ly, 0) * abs(y[site] - (half_Ly + L_y)) + \
                                                            np.heaviside(half_Ly - y[site], 0) * abs(half_Ly - y[site])
    deltaLz = np.heaviside(z[site] - half_Lz, 0) * abs(z[site] - (half_Lz + L_z)) + \
                                                            np.heaviside(half_Lz - z[site], 0) * abs(half_Lz - z[site])

    x, y, z = (x + deltaLx) % L_x, (y + deltaLy) % L_y, (z + deltaLz) % L_z  # Relabel of the operators
    X, Y, Z = np.repeat(x, n_orb), np.repeat(y, n_orb), np.repeat(z, n_orb)  # Vectors for all x, y, z blocks
    X = np.reshape(X, (len(X), 1))                                           # Column vector x
    Y = np.reshape(Y, (len(Y), 1))                                           # Column vector y
    Z = np.reshape(Z, (len(Z), 1))                                           # Column vector z

    # Z2 Invariant calculation
    M = (P @ S @ (X * P) @ (Y * P) @ (Z * P)) + (P @ S @ (Z * P) @ (X * P) @ (Y * P)) + \
        (P @ S @ (Y * P) @ (Z * P) @ (X * P)) - (P @ S @ (X * P) @ (Z * P) @ (Y * P)) - \
        (P @ S @ (Z * P) @ (Y * P) @ (X * P)) - (P @ S @ (Y * P) @ (X * P) @ (Z * P))  # Invariant operator
    
    marker = (8 * pi / 3) * np.imag(np.trace(M[site * n_orb: site * n_orb + n_orb,
                                                       site * n_orb: site * n_orb + n_orb]))  # Local marker 
    return marker

def local_marker_AII(n_orb, L_x, L_y, L_z, x, y, z, P, Q, R, site):

    # Calculates the local CS marker for the specified site
    # L_x, L_y, L_z: Number of lattice sites in each direction
    # n_orb : Number of orbitals
    # n_sites: Number of lattice sites
    # x, y, z: Vectors with the position of the lattice sites
    # P : Valence band projector for the AII path
    # Q :  Valence band projector for the DIII path
    # S: Chiral symmetry operator of the model
    # site: Number of the site we calculate the marker on

    # Contraction with the epsilon tensor
    def contract(P1, P2, P3, P4):
        term = (P1 @ (X * P2) @ (Y * P3) @ (Z * P4)) + (P1 @ (Y * P2) @ (Z * P3) @ (X * P4)) + \
               (P1 @ (Z * P2) @ (X * P3) @ (Y * P4)) - (P1 @ (Y * P2) @ (X * P3) @ (Z * P4)) + \
               -(P1 @ (Z * P2) @ (Y * P3) @ (X * P4)) - (P1 @ (X * P2) @ (Z * P3) @ (Y * P4))
        return term

    # Position operators for the invariant
    half_Lx, half_Ly, half_Lz = np.floor(L_x / 2), np.floor(L_y / 2), np.floor(L_z / 2)
    deltaLx = np.heaviside(x[site] - half_Lx, 0) * abs(x[site] - (half_Lx + L_x)) + \
                                                           np.heaviside(half_Lx - x[site], 0) * abs(half_Lx - x[site])
    deltaLy = np.heaviside(y[site] - half_Ly, 0) * abs(y[site] - (half_Ly + L_y)) + \
                                                           np.heaviside(half_Ly - y[site], 0) * abs(half_Ly - y[site])
    deltaLz = np.heaviside(z[site] - half_Lz, 0) * abs(z[site] - (half_Lz + L_z)) + \
                                                           np.heaviside(half_Lz - z[site], 0) * abs(half_Lz - z[site])

    x, y, z = (x + deltaLx) % L_x, (y + deltaLy) % L_y, (z + deltaLz) % L_z  # Relabel of the operators
    X, Y, Z = np.repeat(x, n_orb), np.repeat(y, n_orb), np.repeat(z, n_orb)  # Vectors for all x, y, z blocks
    X = np.reshape(X, (len(X), 1))                                           # Column vector x
    Y = np.reshape(Y, (len(Y), 1))                                           # Column vector y
    Z = np.reshape(Z, (len(Z), 1))                                           # Column vector z
    S = 1 - (2 * R)                                                          # Chiral symmetry of the path

    # From trivial to DIII
    # M = (Q @ S @ (X * Q) @ (Y * Q) @ (Z * Q)) + (Q @ S @ (Z * Q) @ (X * Q) @ (Y * Q)) + \
    #     (Q @ S @ (Y * Q) @ (Z * Q) @ (X * Q)) - (Q @ S @ (X * Q) @ (Z * Q) @ (Y * Q)) - \
    #     (Q @ S @ (Z * Q) @ (Y * Q) @ (X * Q)) - (Q @ S @ (Y * Q) @ (X * Q) @ (Z * Q))  # Invariant operator
    #
    # marker_DIII = (8 * pi / 3) * np.imag(np.trace(M[site * n_orb: site * n_orb + n_orb,
    #                                               site * n_orb: site * n_orb + n_orb]))  # Local marker
    # Form DIII to AII
    M = (1 / 6) * (-contract(P, Q, P, P) + contract(P, P, Q, P) - contract(P, P, P, Q) + contract(P, Q, Q, Q) +
                   + contract(Q, P, P, P) - contract(Q, Q, Q, P) + contract(Q, Q, P, Q) - contract(Q, P, Q, Q)) + \
        + (1 / 3) * (-2 * contract(P @ Q, P, P, P) - contract(P @ Q, Q, P, P) - contract(P @ Q, P, Q, P) +
                     + contract(P @ Q, Q, Q, P) - contract(P @ Q, P, P, Q) - contract(P @ Q, Q, P, Q) +
                     - contract(P @ Q, P, Q, Q) - 2 * contract(P @ Q, Q, Q, Q) + 2 * contract(P, P, P, P @ Q) +
                     + contract(P, Q, P, P @ Q) + contract(P, P, Q, P @ Q) + contract(P, Q, Q, P @ Q) +
                     + contract(Q, P, P, P @ Q) + contract(Q, Q, P, P @ Q) + contract(Q, P, Q, P @ Q) +
                     + 2 * contract(Q, Q, Q, P @ Q) + 2 * contract(P, P @ Q, P, P) + contract(P, P @ Q, Q, P) +
                     + contract(P, P @ Q, P, Q) + contract(P, P @ Q, Q, Q) + contract(Q, P @ Q, P, P) +
                     + contract(Q, P @ Q, Q, P) + contract(Q, P @ Q, P, Q) + 2 * contract(Q, P @ Q, Q, Q) +
                     - 2 * contract(P, P, P @ Q, P) - contract(P, Q, P @ Q, P) - contract(P, P, P @ Q, Q) +
                     - contract(P, Q, P @ Q, Q) - contract(Q, P, P @ Q, P) - contract(Q, Q, P @ Q, P) +
                     - contract(Q, P, P @ Q, Q) - 2 * contract(Q, Q, P @ Q, Q))

    marker_AII = (8 * pi / 3) * np.imag(np.trace(M[site * n_orb: site * n_orb + n_orb,
                                                 site * n_orb: site * n_orb + n_orb]))  # Local marker

    marker = marker_AII  # + marker_DIII # Total marker

    return marker
# %% Graph for the RPS
def Lattice_graph(L_x, L_y, L_z, n_sites, n_neighbours, x, y, z, matrix_neighbours):

    # Generates the  graph of the RPS
    # L_x, L_y: Dimensions of the RPS grid
    # n_sites: Number of sites in the RPS
    # x: x position of the sites in the RPS
    # y: y position of the sites in the RPS
    # R: Distance cut-off for the hopping amplitudes
    # We can make explicit the couplings at the boundary by changing the specification "Open" to "Closed"
    # (but it becomes a mess)

    cont = 0
    fig = plt.figure()
    axes = fig.add_subplot(111, projection="3d")
    for index1 in range(0, n_sites):
        for index2 in range(1, n_neighbours + 1):
            axes.plot([x[index1], x[matrix_neighbours[index1, index2]]], [y[index1],
                                                                          y[matrix_neighbours[index1, index2]]],
                      [z[index1], z[matrix_neighbours[index1, index2]]],
                      'k', linewidth=1, alpha=0.3)

    axes.plot(x, y, z, '.b', markersize=5)  # Plot of the sites in the RPS
    plt.show()

# %% Debugging
def check_hamiltonians(x, y, z, L_x, L_y, L_z, n_sites, n_orb, H1, H2):
    transx = xtranslation(x, y, z, L_x, L_y, L_z)
    transy = ytranslation(x, y, z, L_x, L_y, L_z)
    transz = ztranslation(x, y, z, L_x, L_y, L_z)
    for index1 in range(n_sites):
        row = n_orb * index1
        column = n_orb * transx[index1]
        check1 = H1[row: row + n_orb, column: column + n_orb]
        check2 = H2[row: row + n_orb, column: column + n_orb]
        check_real = np.sum(abs(np.real(check1 - check2)))
        check_imag = np.sum(abs(np.imag(check1 - check2)))
        if check_real > 0.001:
            print("x_r")
            print(check1 - check2)
            print(check1)
            print("  ")
            print(check2)
            print("-------")
        elif check_imag > 0.001:
            print("x_im")
            print(check1)
            print("  ")
            print(check2)
            print("-------")
    for index1 in range(n_sites):
        row = n_orb * index1
        column = n_orb * transy[index1]
        check1 = H1[row: row + n_orb, column: column + n_orb]
        check2 = H2[row: row + n_orb, column: column + n_orb]
        check_real = np.sum(abs(np.real(check1 - check2)))
        check_imag = np.sum(abs(np.imag(check1 - check2)))
        if check_real > 0.001:
            print("y_r")
            print(check1)
            print("  ")
            print(check2)
            print("-------")
        elif check_imag > 0.001:
            print("y_im")
            print(check1)
            print("  ")
            print(check2)
            print("-------")
    for index1 in range(n_sites):
        row = n_orb * index1
        column = n_orb * transz[index1]
        check1 = H1[row: row + n_orb, column: column + n_orb]
        check2 = H2[row: row + n_orb, column: column + n_orb]
        check_real = np.sum(abs(np.real(check1 - check2)))
        check_imag = np.sum(abs(np.imag(check1 - check2)))
        #print([index1, transz[index1]])
        #print([z[index1], z[transz[index1]]])
        #print(check1)
        #print("-----")
        #print(check2)
        if check_real > 0.001:
            print("z_r")
            print(check1)
            print("  ")
            print(check2)
            print("-------")
        elif check_imag > 0.001:
            print("z_im")
            print(check1)
            print("  ")
            print(check2)
            print("-------")
    for index1 in range(n_sites):
        row = n_orb * index1
        column = row
        check1 = H1[row: row + n_orb, column: column + n_orb]
        check2 = H2[row: row + n_orb, column: column + n_orb]
        check_real = np.sum(abs(np.real(check1 - check2)))
        check_imag = np.sum(abs(np.imag(check1 - check2)))
        if check_real > 0.001:
            print("diag_r")
            print(check1)
            print("  ")
            print(check2)
            print("-------")
        elif check_imag > 0.001:
            print("diag_i")
            print(check_imag)
            print(check1)
            print("  ")
            print(check2)
            print("-------")
    print("done")

