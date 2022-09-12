# %% Various
def RadialHop_AII(r):
    # Calculates the radial hopping strength for two sites at a distance r
    # r : Distance between sites (in units of a)
    C = e  # Normalisation constant for the radial hopping (t=1 for r=a)
    hop_amplitude = C * np.exp(- r)
    return hop_amplitude

def AngularHop_AII(theta, phi, t, lamb):
    # Calculates the angular hopping strength for two sites at an angle theta
    # theta:  Relative angle between sites in z direction
    # theta:  Relative angle between sites in XY cross-section
    hop_matrix = np.zeros((4, 4), complex)  # Block hopping as we have two orbitals
    hop_matrix[0:2, 0:2] = - t * np.eye(2)
    hop_matrix[2:4, 2:4] = t * np.eye(2)
    hop_matrix[0:2, 2:4] = lamb * np.array([[1j * np.cos(theta), 1j * np.exp(-1j * phi) * np.sin(theta)],
                                            [1j * np.exp(1j * phi) * np.sin(theta), -1j * np.cos(theta)]])
    hop_matrix[2:4, 0:2] = lamb * np.array([[1j * np.cos(theta), 1j * np.exp(-1j * phi) * np.sin(theta)],
                                            [1j * np.exp(1j * phi) * np.sin(theta), -1j * np.cos(theta)]])
    return hop_matrix

def Diagonal_AII(M):
    # Calculates the diagonal block associated with the onsite energies and orbital mixing
    # M: Mass parameter of the model
    diag_matrix = np.zeros((4, 4), complex)  # Block hopping as we have two orbitals
    diag_matrix[0, 0] = M
    diag_matrix[1, 1] = M
    diag_matrix[2, 2] = - M
    diag_matrix[3, 3] = - M
    return diag_matrix

def AmorphousHamiltonian_AII_3D(n_sites, n_orb, L_x, L_y, L_z, x, y, z, M, t, lamb, R, boundary):
    # Generates the  AII Hamiltonian for the RPS specified
    # n_sites: Number of sites in the RPS
    # n_orb: Number of orbitals in the model
    # x: x position of the sites in the RPS
    # y: y position of the sites in the RPS
    # z: z position of the sites in the RPS
    # M, t, lamb: Parameters of the AII model
    # R: Distance cut-off for the hopping amplitudes
    # Boundary: String with values "Open" or "Closed" which selects the boundary we want

    Hamiltonian = np.zeros((n_orb * n_sites, n_orb * n_sites), complex)  # Declaration of the Hamiltonian
    cont = 0  # Counter to avoid doing unnecessary iterations in the following loops

    for index1 in range(0, n_sites):
        for index2 in range(cont, n_sites):

            # Auxiliary variables
            r, phi, theta = displacement(x[index1], y[index1], z[index1], x[index2], y[index2], z[index2], L_x, L_y,
                                         L_z, boundary)
            row = index1 * n_orb  # Row in the Hamiltonian taking into account the doubling of the orbitals
            column = index2 * n_orb  # Column in the Hamiltonian taking into account the doubling of the orbitals

            if index1 == index2:
                Hamiltonian[row: row + n_orb, column: column + n_orb] = Diagonal_AII(M)  # Diagonal terms

            elif index1 != index2 and r < R:  # Hopping between sites
                Hamiltonian[row: row + n_orb, column: column + n_orb] = RadialHop_AII(r) * AngularHop_AII(theta, phi, t,
                                                                                                          lamb)
                Hamiltonian[column: column + n_orb, row: row + n_orb] = \
                    np.transpose(np.conj(Hamiltonian[row: row + n_orb, column: column + n_orb]))

        cont = cont + 1  # Update the counter so that we skip through index2 = previous indexes 1
    return Hamiltonian

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

def Hamiltonian3D_AII_BHZ(n_x, n_y, n_z, n_orb, sites, x, y, z, M, t, lamb, boundary):
    # Generates the Hamiltonian for a 3D Z2 insulator with the parameters specified
    # n_x, n_y, n_z: Number of lattice sites in each direction
    # n_orb : Number of orbitals
    # sites: Vector containing the site index of the lattice (remember we enumerate in x direction (0,0)->0 (1,0)->1...)
    # x, y, z: Vectors with the position of the lattice sites
    # t, lamb, M: Parameters of the model
    # boundary: String containing "Open" or "Closed"

    # VARIABLES
    # Lattice definitions
    n_sites = n_x * n_y * n_z  # Number of lattice sites
    n_states = n_sites * n_orb  # Number of basis states
    n_plaquettes = n_x * n_y  # Number of plaquettes
    transx = xtranslation(x, y, z, n_x, n_y, n_z)  # Translation in positive x direction
    transy = ytranslation(x, y, z, n_x, n_y, n_z)  # Translation in positive y direction
    transz = ztranslation(x, y, z, n_x, n_y, n_z)  # Translation in positive z direction
    # Declarations
    H_x = np.zeros((n_states, n_states), dtype='complex_')  # Hamiltonian for the x hoppings
    H_y = np.zeros((n_states, n_states), dtype='complex_')  # Hamiltonian for the y hoppings
    H_z = np.zeros((n_states, n_states), dtype='complex_')  # Hamiltonian for the z hoppings
    H_eps = np.zeros((n_states, n_states), dtype='complex_')  # Hamiltonian for the onsite hoppings
    # Pauli matrices
    sigma_0 = np.eye(2)
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    # Block hoppings in x, y, z and disorder
    blockx = (-t * np.kron(sigma_z, sigma_0) + 1j * lamb * np.kron(sigma_x, sigma_x))  # Block hopping on x direction
    blocky = (-t * np.kron(sigma_z, sigma_0) + 1j * lamb * np.kron(sigma_x, sigma_y))  # Block hopping on y direction
    blockz = (-t * np.kron(sigma_z, sigma_0) + 1j * lamb * np.kron(sigma_x, sigma_z))  # Block hopping on z direction
    blockeps = M * np.kron(sigma_z, sigma_0)  # Onsite hoppings

    for index in range(0, n_sites):
        aux_row = sites[index] * n_orb
        aux_col = transx[index] * n_orb
        if boundary == "Open":  # Takes open boundary conditions on direction x
            if (index + 1) % n_x != 0:
                H_x[aux_row: aux_row + 4, aux_col: aux_col + 4] = blockx  # Assign hopping terms
        else:
            if (index + 1) % n_x == 0:  # Select terms at the x boundary
                H_x[aux_row: aux_row + 4, aux_col: aux_col + 4] = blockx  # Assign hopping terms
            else:  # Rest of the hoppings in x direction
                H_x[aux_row: aux_row + 4, aux_col: aux_col + 4] = blockx  # Assign hopping terms
    for index in range(0, n_sites):
        aux_row = sites[index] * n_orb
        aux_col = transy[index] * n_orb
        if boundary == "Open":  # Takes open boundary conditions on direction y
            if (index % (n_x * n_y) + n_x) < (n_x * n_y):
                H_y[aux_row: aux_row + 4, aux_col: aux_col + 4] = blocky  # Assign hopping terms
        else:
            if (index % (n_x * n_y) + n_x) >= (n_x * n_y):  # Terms at the y boundary
                H_y[aux_row: aux_row + 4, aux_col: aux_col + 4] = blocky  # Assign hopping terms
            else:  # Rest of the hoppings
                H_y[aux_row: aux_row + 4, aux_col: aux_col + 4] = blocky  # Assign hopping terms
    for index in range(0, n_sites):
        aux_row = sites[index] * n_orb
        aux_col = transz[index] * n_orb
        if boundary == "Open":  # Takes open boundary conditions on direction z
            if index + (n_x * n_y) < n_sites:
                H_z[aux_row: aux_row + 4, aux_col: aux_col + 4] = blockz  # Assign hopping terms
        else:
            H_z[aux_row: aux_row + 4, aux_col: aux_col + 4] = blockz  # Assign hopping terms
    for index in range(0, n_sites):
        aux = sites[index] * n_orb
        H_eps[aux: aux + 4, aux: aux + 4] = blockeps  # Assign hopping terms
    H = H_x + H_y + H_z
    H = H + np.conj(np.transpose(H)) + H_eps  # Total hamiltonian

    return H

def Hamiltonian3D(n_x, n_y, n_z, n_orb, sites, x, y, z, m, t1, t2, lamb, mu, boundary):
    # Generates the Hamiltonian for a 3D insulator with the parameters specified
    # t2 = 0  we are in the DIII class (S = - sigma0 x sigma_y), if t2 is non-zero we are in the AII class
    # n_x, n_y, n_z: Number of lattice sites in each direction
    # n_orb : Number of orbitals
    # sites: Vector containing the site index of the lattice (remember we enumerate in x direction (0,0)->0 (1,0)->1...)
    # x, y, z: Vectors with the position of the lattice sites
    # m, t1, t2, lamb, mu: Parameters of the model
    # boundary: String containing "Open" or "Closed"

    # VARIABLES
    # Lattice definitions
    n_sites = n_x * n_y * n_z  # Number of lattice sites
    n_states = n_sites * n_orb  # Number of basis states
    transx = xtranslation(x, y, z, n_x, n_y, n_z)  # Translation in positive x direction
    transy = ytranslation(x, y, z, n_x, n_y, n_z)  # Translation in positive y direction
    transz = ztranslation(x, y, z, n_x, n_y, n_z)  # Translation in positive z direction
    # Declarations
    H_x = np.zeros((n_states, n_states), dtype='complex_')  # Hamiltonian for the x hoppings
    H_y = np.zeros((n_states, n_states), dtype='complex_')  # Hamiltonian for the y hoppings
    H_z = np.zeros((n_states, n_states), dtype='complex_')  # Hamiltonian for the z hoppings
    H_eps = np.zeros((n_states, n_states), dtype='complex_')  # Hamiltonian for the onsite hoppings
    H_disorder = np.zeros((n_states, n_states), dtype='complex_')  # Hamiltonian for the onsite hoppings
    # Pauli matrices
    sigma_0 = np.eye(2)
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    # Block hoppings in x, y, z and disorder
    blockx = 0.5 * (1j * lamb * np.kron(sigma_x, sigma_x) + t2 * np.kron(sigma_x, sigma_y) - t1 * np.kron(sigma_0,
                                                                                                          sigma_z))  # Block hopping on x direction
    blocky = 0.5 * (1j * lamb * np.kron(sigma_y, sigma_x) + t2 * np.kron(sigma_y, sigma_y) - t1 * np.kron(sigma_0,
                                                                                                          sigma_z))  # Block hopping on x direction
    blockz = 0.5 * (1j * lamb * np.kron(sigma_z, sigma_x) + t2 * np.kron(sigma_z, sigma_y) - t1 * np.kron(sigma_0,
                                                                                                          sigma_z))  # Block hopping on x direction
    blockeps = - m * np.kron(sigma_0, sigma_z)  # Diagonal block

    # Hamiltonian

    # x hoppings
    for index in range(0, n_sites):
        aux_row = sites[index] * n_orb
        aux_col = transx[index] * n_orb
        if boundary == "Open":  # Takes open boundary conditions on direction x
            if (index + 1) % n_x != 0:
                H_x[aux_row: aux_row + 4, aux_col: aux_col + 4] = blockx  # Assign hopping terms
        else:
            H_x[aux_row: aux_row + 4, aux_col: aux_col + 4] = blockx  # Assign hopping terms

    # y hoppings
    for index in range(0, n_sites):
        aux_row = sites[index] * n_orb
        aux_col = transy[index] * n_orb
        if boundary == "Open":  # Takes open boundary conditions on direction y
            if (index % (n_x * n_y) + n_x) < (n_x * n_y):
                H_y[aux_row: aux_row + 4, aux_col: aux_col + 4] = blocky  # Assign hopping terms
        else:
            H_y[aux_row: aux_row + 4, aux_col: aux_col + 4] = blocky  # Assign hopping terms

    # z hoppings
    for index in range(0, n_sites):
        aux_row = sites[index] * n_orb
        aux_col = transz[index] * n_orb
        if boundary == "Open":  # Takes open boundary conditions on direction z
            if index + (n_x * n_y) < n_sites:
                H_z[aux_row: aux_row + 4, aux_col: aux_col + 4] = blockz  # Assign hopping terms
        else:
            H_z[aux_row: aux_row + 4, aux_col: aux_col + 4] = blockz  # Assign hopping terms

    # onsite hoppings
    for index in range(0, n_sites):
        aux = sites[index] * n_orb
        disorder = mu * ((-1) ** (np.random.randint(2, size=1))) * np.random.rand(1) * np.kron(sigma_0, sigma_z)
        H_eps[aux: aux + 4, aux: aux + 4] = blockeps  # Assign hopping terms
        H_disorder[aux: aux + 4, aux: aux + 4] = disorder

    H = H_x + H_y + H_z
    H = H + np.conj(np.transpose(H)) + H_eps + H_disorder  # Total hamiltonian

    return H

def AmorphousHamiltonian3D(n_sites, n_orb, L_x, L_y, L_z, x, y, z, M, t1, t2, lamb, R, boundary):
    # Generates the Hamiltonian for a 3D Z2 insulator with the parameters specified
    # t2 = 0  we are in the DIII class (S = - sigma0 x sigma_y), if t2 is non-zero we are in the AII class
    # n_sites: Number of sites in the RPS
    # n_orb: Number of orbitals in the model
    # x: x position of the sites in the RPS
    # y: y position of the sites in the RPS
    # z: z position of the sites in the RPS
    # M, t1, t2, lamb: Parameters of the model
    # R: Distance cut-off for the hopping amplitudes
    # Boundary: String with values "Open" or "Closed" which selects the boundary we want

    Hamiltonian = np.zeros((n_orb * n_sites, n_orb * n_sites), complex)  # Declaration of the Hamiltonian
    cont = 0  # Counter to avoid doing unnecessary iterations in the following loops

    for index1 in range(0, n_sites):
        for index2 in range(cont, n_sites):

            # Auxiliary variables
            r, phi, theta = displacement(x[index1], y[index1], z[index1], x[index2], y[index2], z[index2], L_x, L_y,
                                         L_z, boundary)
            row = index1 * n_orb  # Row in the Hamiltonian taking into account the doubling of the orbitals
            column = index2 * n_orb  # Column in the Hamiltonian taking into account the doubling of the orbitals

            if index1 == index2:
                Hamiltonian[row: row + n_orb, column: column + n_orb] = Diagonal(M)  # Diagonal terms

            elif index1 != index2 and r < R:  # Hopping between sites
                Hamiltonian[row: row + n_orb, column: column + n_orb] = RadialHop(r) * AngularHop(theta, phi, t1, t2,
                                                                                                  lamb)
                Hamiltonian[column: column + n_orb, row: row + n_orb] = \
                    np.transpose(np.conj(Hamiltonian[row: row + n_orb, column: column + n_orb]))

        cont = cont + 1  # Update the counter so that we skip through index2 = previous indexes 1
    return Hamiltonian

def local_marker_open(n_orb, n_sites, n_states, x, y, z, eigenstates, chiral_symmetry, n_particles, sites):
    # Operator definitions for the Z2 invariant
    X, Y, Z = np.zeros((n_states, n_states)), np.zeros((n_states, n_states)), np.zeros(
        (n_states, n_states))  # X,Y,Z Operators
    S = np.zeros((n_states, n_states), complex)  # Chiral symmetry operator
    CS_marker = np.zeros((len(sites),))

    for index in range(0, n_sites):
        X[index * n_orb: index * n_orb + n_orb, index * n_orb: index * n_orb + n_orb] = x[index] * np.eye(n_orb)
        Y[index * n_orb: index * n_orb + n_orb, index * n_orb: index * n_orb + n_orb] = y[index] * np.eye(n_orb)
        Z[index * n_orb: index * n_orb + n_orb, index * n_orb: index * n_orb + n_orb] = z[index] * np.eye(n_orb)
        S[index * n_orb: index * n_orb + n_orb, index * n_orb: index * n_orb + n_orb] = chiral_symmetry

    # Valence band projector
    U = np.zeros((n_states, n_states), complex)
    U[:, 0: n_particles] = eigenstates[:, 0: n_particles]
    P = U @ np.conj(np.transpose(U))

    # Z2 Invariant calculation
    M = (P @ S @ X @ P @ Y @ P @ Z @ P) + (P @ S @ Z @ P @ X @ P @ Y @ P) + (P @ S @ Y @ P @ Z @ P @ X @ P) - \
        (P @ S @ X @ P @ Z @ P @ Y @ P) - (P @ S @ Z @ P @ Y @ P @ X @ P) - (P @ S @ Y @ P @ X @ P @ Z @ P)

    for index in range(len(sites)):
        CS_marker[index] = (8 * pi / 3) * np.imag(np.trace(M[sites[index] * n_orb: sites[index] * n_orb + n_orb,
                                                           sites[index] * n_orb: sites[index] * n_orb + n_orb]))

    return CS_marker