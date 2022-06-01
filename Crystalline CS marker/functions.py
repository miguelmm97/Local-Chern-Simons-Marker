# FUNCTIONS

import numpy as np
from numpy import pi

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

def Hamiltonian3D_DIII_FB(n_x, n_y, n_z, n_orb, sites, x, y, z, t, lamb, lamb_z, eps, flux, boundary):
    # Generates the Hamiltonian for a 3D Z2 insulator with the parameters specified
    # n_x, n_y, n_z: Number of lattice sites in each direction
    # n_orb : Number of orbitals
    # sites: Vector containing the site index of the lattice (remember we enumerate in x direction (0,0)->0 (1,0)->1...)
    # x, y, z: Vectors with the position of the lattice sites
    # t, lamb, lamb_z, flux: Parameters of the model
    # open_boundary: String containing "True" or "False" (DISCLAIMER: When the boundaries are closed, we still add
    # a non-vanishing built-in flux to the model!!!)

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
    blockx = +((1j * lamb / 2 * np.kron(sigma_y, sigma_z)) - t * np.kron(sigma_0,
                                                                        sigma_x))  # Block hopping on x direction
    blocky = -(-1j * lamb / 2 * np.kron(sigma_x, sigma_z)) - t * np.kron(sigma_0,
                                                                        sigma_x)  # Block hopping on y direction
    blockz = t * np.kron(sigma_0, sigma_x) - lamb_z * 1j / 2 * np.kron(sigma_0,
                                                                        sigma_y)  # Block hopping on z direction
    blockeps = eps * np.kron(sigma_0, sigma_x)
    # Peierls substitution factors
    peierlsx = np.exp(
        - 2 * pi * 1j * flux * y / n_plaquettes)  # Peierls phases for implementing the flux in the lattice model
    peierlsx_boundary = np.exp(- 2 * pi * 1j * (
            1 - n_x) * flux * y / n_plaquettes)  # Peierls phases for implementing the flux in the lattice model
    peierlsy = np.exp(- 2 * pi * 1j * (1 - n_y) * flux * (x + 1) * (n_y - 2) / ((n_y - 1) * n_plaquettes))


    for index in range(0, n_sites):
        aux_row = sites[index] * n_orb
        aux_col = transx[index] * n_orb
        if boundary == "Open":  # Takes open boundary conditions on direction x
            if (index + 1) % n_x != 0:
                H_x[aux_row: aux_row + 4, aux_col: aux_col + 4] = blockx * peierlsx[index]  # Assign hopping terms
        else:
            if (index + 1) % n_x == 0:  # Select terms at the x boundary
                H_x[aux_row: aux_row + 4, aux_col: aux_col + 4] = blockx * peierlsx_boundary[index]  # Assign hopping terms
            else:  # Rest of the hoppings in x direction
                H_x[aux_row: aux_row + 4, aux_col: aux_col + 4] = blockx * peierlsx[index]  # Assign hopping terms
    for index in range(0, n_sites):
        aux_row = sites[index] * n_orb
        aux_col = transy[index] * n_orb
        if boundary == "Open":  # Takes open boundary conditions on direction y
            if (index % (n_x * n_y) + n_x) < (n_x * n_y):
                H_y[aux_row: aux_row + 4, aux_col: aux_col + 4] = blocky  # Assign hopping terms
        else:
            if (index % (n_x * n_y) + n_x) >= (n_x * n_y):  # Terms at the y boundary
                H_y[aux_row: aux_row + 4, aux_col: aux_col + 4] = blocky * peierlsy[index]  # Assign hopping terms
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
    # Added Peierls factor to realise a built-in flux in the periodic lattice
    # if open_boundary == "True":
    #     for index in range(0, n_z):
    #         H[(n_x * n_y) * index: (n_x * n_y) * index + 4, ((n_x * n_y) - n_x) + (n_x * n_y) * index: ((n_x * n_y) - n_x) \
    #           + (n_x * n_y) * index + 4] = blocky * np.exp(- 2 * pi * 1j * (1 - n_y) * flux
    #                                                        * n_x * (n_y - 2) / ((n_y - 1) * n_plaquettes))

    return H

def Hamiltonian3D_DIII(n_x, n_y, n_z, n_orb, sites, x, y, z, m, boundary):
    # Generates the Hamiltonian for a 3D Z2 insulator with the parameters specified
    # n_x, n_y, n_z: Number of lattice sites in each direction
    # n_orb : Number of orbitals
    # sites: Vector containing the site index of the lattice (remember we enumerate in x direction (0,0)->0 (1,0)->1...)
    # x, y, z: Vectors with the position of the lattice sites
    # t, lamb, lamb_z, flux: Parameters of the model
    # open_boundary: String containing "True" or "False" (DISCLAIMER: When the boundaries are closed, we still add
    # a non-vanishing built-in flux to the model!!!)

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
    # Pauli matrices
    sigma_0 = np.eye(2)
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    # Block hoppings in x, y, z and disorder
    blockx = 0.5 * (1j * np.kron(sigma_x, sigma_x) - np.kron(sigma_0, sigma_z))  # Block hopping on x direction
    blocky = 0.5 * (1j * np.kron(sigma_y, sigma_x) - np.kron(sigma_0, sigma_z))  # Block hopping on x direction
    blockz = 0.5 * (1j * np.kron(sigma_z, sigma_x) - np.kron(sigma_0, sigma_z))  # Block hopping on x direction
    blockeps = - m * np.kron(sigma_0, sigma_z)


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
                H_x[aux_row: aux_row + 4, aux_col: aux_col + 4] = blockx   # Assign hopping terms
    for index in range(0, n_sites):
        aux_row = sites[index] * n_orb
        aux_col = transy[index] * n_orb
        if boundary == "Open":  # Takes open boundary conditions on direction y
            if (index % (n_x * n_y) + n_x) < (n_x * n_y):
                H_y[aux_row: aux_row + 4, aux_col: aux_col + 4] = blocky  # Assign hopping terms
        else:
            if (index % (n_x * n_y) + n_x) >= (n_x * n_y):  # Terms at the y boundary
                H_y[aux_row: aux_row + 4, aux_col: aux_col + 4] = blocky   # Assign hopping terms
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



