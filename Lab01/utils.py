import copy


def determinant3(matrix: list[list]):
    return (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
            matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
            matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))


def determinant2(matrix: list[list]):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def trace(matrix: list[list]):
    return matrix[0][0] + matrix[1][1] + matrix[2][2]


def vector_norm(matrix: list):
    return (matrix[0] ** 2 + matrix[1] ** 2 + matrix[2] ** 2)**0.5


def transpose(matrix: list[list]):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]


def matrix_vector_multiplication(matrix_a: list[list], vector: list[list]):
    return [sum(matrix_a[i][j] * vector[j] for j in range(len(vector))) for i in range(len(matrix_a))]


def replace_column(matrix_a, matrix_b, col):
    new_matrix = copy.deepcopy(matrix_a)
    for i in range(len(matrix_b)):
        new_matrix[i][col] = matrix_b[i]
    return new_matrix


def cofactor(matrix_a):
    matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(3):
        for j in range(3):
            minor = [row[:j] + row[j+1:] for k, row in enumerate(matrix_a) if k != i]
            matrix[i][j] = ((-1) ** (i + j)) * determinant2(minor)
    return matrix


def adjugate(matrix_a):
    cof = cofactor(matrix_a)
    return transpose(cof)


def inverse(matrix_a):
    det_a = determinant3(matrix_a)
    if det_a == 0:
        raise ValueError("Matrix is not invertible (det(A) = 0)")
    adj_a = adjugate(matrix_a)
    return [[adj_a[i][j] / det_a for j in range(3)] for i in range(3)]