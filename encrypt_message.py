import numpy as np

def encrypt_message(message, key_matrix):
    vector = np.array([ord(char) for char in message])

    eigenvalues, P = np.linalg.eig(key_matrix)
    D = np.diag(eigenvalues)
    P_inverse = np.linalg.inv(P)

    encrypted_vector = P @ D @ P_inverse @ vector
    return encrypted_vector


def decrypt_message(encrypted_vector, key_matrix):
    eigenvalues, P = np.linalg.eig(key_matrix) # перетворюємо повідомлення в ASCII код
    P_inverse = np.linalg.inv(P)

    D_inverse = np.diag(1 / eigenvalues)

    decrypted_vector = P @ D_inverse @ P_inverse @ encrypted_vector # шифрування

    decrypted_message = ''.join([chr(int(round(num))) for num in decrypted_vector.real])

    return decrypted_message

first_message = "Hello, World!"

key_matrix = np.random.rand(len(first_message), len(first_message))

encrypted_vector = encrypt_message(first_message, key_matrix)
decrypted_message = decrypt_message(encrypted_vector, key_matrix)

print("Message:", first_message)
print("Encrypted Vector:", encrypted_vector)
print("Decrypted Message:", decrypted_message)
