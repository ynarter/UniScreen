import numpy as np

# Example input
matrix = np.random.rand(5, 4)  # N x 4 matrix (e.g., N=5)
vector = np.ones(5)     # Vector with size N

# Multiply each row of the matrix by the corresponding element in the vector
result = (matrix.T * vector).T  # Transpose, multiply, and transpose back

# Flatten the result into a vector of size 4N
flattened_result = result.flatten()

print("Original Matrix:\n", matrix.T)
print("Vector:\n", vector)
print("Resulting Flattened Vector:\n", flattened_result)

matrix = np.zeros((5, 4 * 5))

# Assign 1s in the appropriate positions
for i in range(5):
    matrix[i, 4 * i : 4 * (i + 1)] = 1

print(matrix)