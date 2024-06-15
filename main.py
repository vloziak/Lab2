import numpy as np

def find_eigenvalues(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    return eigenvalues

def find_eigenvectors(matrix):
    eigenvectors = np.linalg.eig(matrix)
    return eigenvectors

A = np.array([[-5, 0, 3],
              [-6, 1, 3],
              [-6, 0, 4]])

eigenvalues = find_eigenvalues(A)
eigenvectors = find_eigenvectors(A)

print("Власні значення:", eigenvalues)
print("Власні вектори:", eigenvectors)