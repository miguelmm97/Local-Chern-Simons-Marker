#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 10:48:54 2022

@author: jensba
"""

import numpy as np
from scipy import sparse
import itertools
from dataclasses import dataclass

def bin_to_n(s, L):
    """
    Given an integer s representing a basis state in an L site system,
    return the occupation number representation as an array.
    If the array is n = [n0,n1,...,n(L-1)] then
    s = n0*2^0 + n1*2^2 + ... + n(L-1)*2^{L-1}
    Note that binary representation of an integer is in the opposite order.

    Parameters
    ----------
    s : integer
    L : number of sites
    
    Returns
    ----------
    n : numpy array of occupations [n0,n1,...,n(L-1)] with ni = 0 or 1
    
    Example
    ----------
    >>> bin_to_n(10, 4)
    array([0, 1, 0, 1])

    Explanation
    ----------
    bin(s) takes the integer s to a binary string, whose first to letters are a "decoration" and we don't need them
    so we tak the [2:]. We fill with zeros at the start of the binary string to acount for all sites with zfill(L) and
    then we reverse the binary string, in order to read the physical sites from left to right.

    """

    return np.array([int(x) for x in reversed(bin(s)[2:].zfill(L))])

def n_to_bin(n):
    """
    Given an occupation number representation of a state n, 
    return the integer corresponding to its binary representation.
    If the array is n = [n0,n1,...,n(L-1)] then
    s = n0*2^0 + n1*2^2 + ... + n(L-1)*2^{L-1}
    Note that binary representation of an integer is in the opposite order.

    Parameters
    ----------
    n : numpy array of occupations [n0,n1,...,n(L-1)] with ni = 0 or 1
    
    Returns
    ----------
    s : integer

    Example
    ----------
    >>> n_to_bin([0, 1, 0, 1])
    10

    Explanation
    ----------
    We take the array of occupation sites n and reverse it. int(reversed(n), 2) tells python that n is in a binary base
    and gives back the correspoinding integer.


    """
    
    return int(''.join(str(x) for x in reversed(n)), 2)

def spectrum(H):
    # Calculates the spectrum a of the given Hamiltonian
    # H: Hamiltonian for the model
    # n_particles: Number of particles we want (needed for the projector)

    energy, eigenstates = np.linalg.eigh(H)  # Diagonalise H
    idx = energy.argsort()  # Indexes from lower to higher energy
    energy = energy[idx]  # Ordered energy eigenvalues
    eigenstates = eigenstates[:, idx]  # Ordered eigenstates

    return energy, eigenstates


@dataclass
class model:
    """ spinless 1D fermions """
    t: float     # nearest neighbor hopping t(c^dagger_i c_{i+1} + h.c)
    V: float     # nearest neighbor density-density interaction Vc^dagger_ic_i
    mu: float    # chemical potential -mu (c^\dagger_i c_i -1/2)
    Delta: float # Pairing potential Delta c_i c_{i+1}
    L: int       # Number of sites
   
    
    def select_parity(self):
        """
        Separates the full Hilbert space into even and odd sectors.
        The a integer runs over all integers, while the rp = 0,...,2^{L-1}-1
        indices the basis states of the even sector, and rm similarily the odd sector
        The mapping between the two set if integers is given by a_to_rp and rp_to_a
        and similar for a_to_rm and rm_to_a.

        Returns
        -------
        None.

        Explanation
        --------
        We take the basis states 0, 2^L, and write them in occupation number representation with bin_to_a. If there is
        an even number of occupied sites then we are in the even subspace, if the number is odd we are in the odd subspace.
        We save these two subspace by means of two strings: a_to_rp/rm gives the binary representation of a state within
        the even/odd subspace, and rp/rm_to_a gives the binary representation in the full Hilbert space of a state a with
        binary representation in rp(rm in the even/odd subspace.

        """
        L = self.L
        a_to_rp = {}
        a_to_rm = {}
        rp_to_a = {}
        rm_to_a = {}
        rp = 0; rm = 0
        for a in range(2 ** L):
            n_a = bin_to_n(a,L)
            if np.sum(n_a) % 2 == 0:
                a_to_rp[a] = rp
                rp_to_a[rp] = a
                rp += 1
            else:
                a_to_rm[a] = rm
                rm_to_a[rm] = a
                rm += 1
                

        self.psi2rp = a_to_rp
        self.rp2psi = rp_to_a
        self.psi2rm = a_to_rm
        self.rm2psi = rm_to_a

    def select_sector(self):
        """
        Separates the Fock space in the different fixed number sectors.

        Explanation:
        -----------
        We split the Fock space in the different fixed number sectors. We store the mapping from the 2 ** L state basis
        {psi} to the subspace with fixed N basis in the dictionaries psi_to_nsec, nsec_to_psi, where the key to access
        the list containing the mapping is just the number of particles in the sector

        Example:
        -------
        self.psi_to_nsec[3] = [0, 1, 2, 3, 4, 5, 6, ...] basis in the n=3 sector
        self.nsec_to_psi[3] = [10, 11, 12, .....] basis in the 2 ** L Fock space
        """
        L = self.L
        psi_to_nsec = {}  # Dictionary of lists that map psi to the basis in a fixed number sector
        nsec_to_psi = {}  # Dictionary of lists that map the basis in a fixed number sector to psi
        dim = {}

        for i in range(self.L + 1):
            psi_to_nsecL = []  # List of the mapping from psi to the fixed number sector n=i
            nsec_to_psiL = []  # List of the mapping from the fixed number sector n=i tp psi
            state = 0  # Counts how many states we have in the sector

            for psi in range(2 ** L):
                n_psi = bin_to_n(psi, L)  # Occupation number representation of psi
                # Sector with n=i
                if np.sum(n_psi) == i:
                    psi_to_nsecL.append(state)
                    nsec_to_psiL.append(psi)
                    state += 1

            psi_to_nsec[i] = psi_to_nsecL
            nsec_to_psi[i] = nsec_to_psiL
            dim[i] = state

        self.psi2nsec = psi_to_nsec
        self.nsec2psi = nsec_to_psi
        self.dim = dim

    def calc_Hamiltonian(self, parity=None, sector=None, bc='periodic'):
        """
        Calculates the Hamiltonian either in a parity sector or a fixed number sector.

        Parameters
        ----------
        parity: {string, optional} Chooses the parity sector that is to be calculated. It can be either 'even' or 'odd'
        sector: {int, optional} Chooses the number sector. It must be an integer number indicating the fermion number in the chain.
        bc: {string, optional} Sets the boundary condition. Can be either 'periodic' or 'open'. The default is 'periodic'.

        Raises
        ------
        Errors if the boundary, sector or parity are not well-defined.
        Errors if both parity and number sector are passed to the function.

        Returns
        -------
        H : np.array
            The Hamiltonian in the given sector with given boundary conditions.

        """
        # Definitions
        L = self.L

        if parity is not None and sector is not None:
            raise ValueError('Either parity or number sector, both is too much :) ')

        # Selecting states in the parity sector
        if parity is not None:
            dim = 2 ** (L - 1)
            H = np.zeros((dim, dim))
            self.select_parity()
            if parity == 'even':
                block_to_psi = self.rp2psi
                psi_to_block = self.psi2rp
            elif parity == 'odd':
                block_to_psi = self.rm2psi
                psi_to_block = self.psi2rm
            else:
                raise ValueError('parity must be "even" or "odd"')

        # Selecting states in a fixed number sector
        if sector is not None:
            if isinstance(sector, int):
                if self.Delta != 0:
                    raise ValueError('there must be no pairing if we want particle number conservation!')
                # dim = int(factorial(int(sector + L - 1)) / (factorial(int(sector)) * factorial(int(L - 1))))
                self.select_sector()
                dim = self.dim[sector]
                H = np.zeros((dim, dim))
                psi_to_block = self.psi2nsec[sector]
                block_to_psi = self.nsec2psi[sector]
            else:
                raise ValueError('sector must be an integer number')

        # Selecting boundary conditions
        if bc == 'periodic':
            Lhop = L
        elif bc == 'open':
            Lhop = L - 1
        else:
            raise ValueError('boundary condition must be "periodic" or "open"')

        # Construction of the hamiltonian
        for r in range(dim):
            psi = block_to_psi[r]
            n = bin_to_n(psi, L)

            # Diagonal terms (density-density and Anderson disorder
            H[r, r] += np.dot(n - 0.5, self.mu * np.ones(L)) + self.V * np.dot(n[:Lhop], np.roll(n, -1)[:Lhop])

            # Nearest neighbour terms
            for i in range(Lhop):
                j = np.mod(i + 1, L)  # j is either i+1 or 0

                # Hopping
                try:
                    phi, phase = self.hopping(n, i, j)  # State connected by hopping in the binary representation
                    s = psi_to_block[block_to_psi.index(phi)]  # Binary rep of phi in the symmetry sector
                    ht = - self.t * phase  # Hopping times the exponent of the hopping term
                    H[s, r] += ht
                    H[r, s] += np.conjugate(ht)
                except TypeError:
                    pass

                # Pairing term
                try:
                    phi, h = self.pairing(n, i, j)
                    s = psi_to_block[block_to_psi.index(phi)]
                    hp = self.Delta * h
                    H[s, r] += hp
                    H[r, s] += np.conjugate(hp)
                except TypeError:
                    pass

        return H

    def hopping(self, n, i, j):
        """

        Parameters
        ----------
        n: Occupation number representation of an initial state
        i: Site to which we hopp
        j: Site from which we hopp

        Returns
        -------
        [0]: State connected to n by the hopping term (in the full binary representation)
        [1]: Exponent that gives us the sign depending on the commutations that we had to do

        Explanation
        -------
        If the hopping is onsite, we give back the same state. If not, we need site i to be empty and site j to be occupied.
        The state we get back is, in the full binary description, just setting i to 1 and j to 0 (a + 2**i - 2**j),
        times the exponent (which one can see by calculating the hopping term analytically)
        """
        a = n_to_bin(n)
        if i == j and n[i] == 1:
            return a, 1
        elif n[i] == 0 and n[j] == 1:
            exponent = np.sum(n[min(i, j)+1:max(i, j)])
            return a + 2 ** i - 2 ** j, (-1)**exponent
        else:
            return None

    def pairing(self, n, i, j):
        """

        Parameters
        ----------
        n: Occupation number representation of an initial state
        i: Site we pair with site j
        j: Site we want to pair

        Returns
        -------
        [0]: State connected to n by the hopping term (in the full binary representation)
        [1]: Exponent that gives us the sign depending on the commutations that we had to do

        Explanation
        -------
        Same flavour as with the hopping term, just different analytical implementation.


        """

        if i != j and (n[i]== 1 and n[j] == 1):
            a = n_to_bin(n)
            exponent = np.sum(n[min(i,j):max(i,j)])
            return a - 2**i - 2**j, np.sign(i-j)*(-1)**exponent
        else:
            return None

    def calc_opdm_operator(self, parity='even'):
        """
        calculates the matrix rho_ij = <r|c_i^dagger c_j|s>
        and <r|c_i c_j|s> in subspace of even or odd parity.
        rho['even/odd'][i,j] = rho_ij
        with rho_ij a sparse matrix.

        Parameters
        ----------
        parity : string, optional
            The parity of the basis for rho. The default is 'even'.


        Returns
        -------
        None. But sets self.rho to rho.

        """

        L = self.L
        dim = 2 ** (L - 1)

        rho = {}
        rho['eh'] = {}
        rho['hh'] = {}
        # Initializing rho_eh operator
        for i, j in itertools.product(range(L), range(L)):
            rho['eh'][i, j] = sparse.lil_matrix((dim, dim))
            rho['hh'][i, j] = sparse.lil_matrix((dim, dim))

        if parity == 'even':
            r_to_a = self.rp2psi
            a_to_r = self.psi2rp
        elif parity == 'odd':
            r_to_a = self.rm2psi
            a_to_r = self.psi2rm
        else:
            raise ValueError('parity must be "even" or "odd"')

        for r in range(dim):
            a = r_to_a[r]
            n = bin_to_n(a, L)
            # mprint(a,r,bin_to_n(a,L))
            for i, j in itertools.product(range(L), range(L)):
                try:
                    b, alpha = self.hopping(n, i, j)
                    rho['eh'][i, j][a_to_r[b], r] = alpha
                except TypeError:
                    pass

                try:
                    b, alpha = self.pairing(n, i, j)
                    rho['hh'][i, j][a_to_r[b], r] = alpha
                except TypeError:
                    pass

        # Convert sparse lil matrices to csr
        self.rho = {}
        self.rho['eh'] = {}
        self.rho['hh'] = {}
        for key in rho['eh']:
            self.rho['eh'][key] = rho['eh'][key].tocsr()
            self.rho['hh'][key] = rho['hh'][key].tocsr()

        return None

    def calc_opdm_from_psi(self, psi, parity='even'):
        """
        Calculates the one-particle-density matrix from a state psi.

        rho_opdm =  [[ <psi|c^\dagger_i c_j|psi> , <psi|c^\dagger_i c^\dagger_j|psi>],
                     [ <psi|c_i c_j|psi>,<psi|c_i c^\dagger_j|psi> ]]

        Parameters
        ----------
        psi : numpy arrary
            The state for which we want the opdm.
        parity : TYPE, optional
            the parity of the subspace in which the state lives. The default is 'even'.

        Returns
        -------
        rho_opdm : numpy array 2L x 2L
            the opdm.
        """

        if not hasattr(self, 'rho'):
            self.calc_opdm_operator(parity)

        L = self.L
        rho_opdm = np.zeros((2 * L, 2 * L))

        for i, j in itertools.product(range(L), range(L)):
            rho_opdm[i, j] = np.dot(psi.conjugate(), (self.rho['eh'][i, j] * psi))
            rho_opdm[i + L, j] = np.dot(psi.conjugate(), (self.rho['hh'][i, j] * psi))
        rho_opdm[L: 2 * L, L: 2 * L] = np.eye(L) - rho_opdm[:L, :L].T
        rho_opdm[:L, L:2 * L] = rho_opdm[L: 2 * L, :L].T.conjugate()

        return rho_opdm


