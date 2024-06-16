import numpy as np

def find_eigenvectors_eigenvalues(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return eigenvalues, eigenvectors
def checking_eigenvectors(matrix, eigenvalues, eigenvectors):
    print("Перевірка рівності A⋅v = λ⋅v:")

    for i in range(len(eigenvalues)):
        eigenvalue = eigenvalues[i]
        eigenvector = eigenvectors[:, i]
        A_v = np.dot(matrix, eigenvector)
        λ_v = eigenvalue * eigenvector

        print(f"\nВласне значення λ = {eigenvalue}")
        print(f"Власний вектор v = {eigenvector}")
        print(f"A⋅v = {A_v}")
        print(f"λ⋅v = {λ_v}")

        if np.allclose(A_v, λ_v):
            print(f"Рівність A⋅v = λ⋅v виконується для λ = {eigenvalue}\n")
        else:
            print(f"Рівність A⋅v = λ⋅v НЕ виконується для λ = {eigenvalue}\n")


A = np.array([[-5, 0, 3],
              [-6, 1, 3],
              [-6, 0, 4]])

eigenvalues,eigenvectors = find_eigenvectors_eigenvalues(A)

print("Власні значення:", eigenvalues)
print("Власні вектори:", eigenvectors)
checking_eigenvectors(A, eigenvalues, eigenvectors)