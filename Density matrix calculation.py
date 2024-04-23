import numpy as np
from scipy.linalg import expm

def evolve_density_matrix(rho_0, X, Z, t):
    """
    Evolve the density matrix rho according to the Lindbladian equation.
    :param rho_0: Initial density matrix
    :param X: Pauli X matrix
    :param Z: Pauli Z matrix
    :param t: Time to evolve
    :return: Evolved density matrix at time t
    """
    # Define the Lindbladian superoperator
    def L(rho):
        return -1j * (np.dot(X, rho) - np.dot(rho, X)) + 0.5 * (np.dot(Z, np.dot(rho, Z)) - rho)

    # Since the Lindbladian is time-independent, we can use matrix exponential
    # We need to vectorize the density matrix to use the expm function
    rho_vec = rho_0.flatten()
    L_matrix = np.kron(np.eye(2), -1j*X + 0.5*Z) - np.kron(np.conj(X), 1j*np.eye(2)) + 0.5*np.kron(np.eye(2), Z) - 0.5*np.kron(np.conj(Z), np.eye(2))
    rho_t_vec = expm(L_matrix * t) @ rho_vec

    # Reshape the vectorized density matrix back to its matrix form
    rho_t = rho_t_vec.reshape((2, 2))
    return rho_t

# Define Pauli matrices
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])

# Initial state |0><0|
rho_0 = np.array([[1, 0], [0, 0]])

# Evolve the density matrix for time t=2
rho_t = evolve_density_matrix(rho_0, X, Z, 2)

# Print the resulting density matrix
print('The density matrix at time t=2 is:')
print(rho_t)