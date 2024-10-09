from utils import determinant3, replace_column, inverse, matrix_vector_multiplication


def parse_equations(filename) -> (list, list):
    left = []
    right = []

    with open(filename, 'r') as file:
        for line in file:
            equation = line.replace(" ", "").split('=')
            left_side = equation[0]
            right_side = int(equation[1])

            base_value = 1
            coefficients = [0, 0, 0]
            terms = left_side.replace('-', ' -').replace('+', ' +').split()
            for term in terms:
                if 'x' in term:
                    try:
                        coefficients[0] = int(term.replace('x', ''))
                    except ValueError:
                        coefficients[0] = -base_value if '-' in term else base_value

                elif 'y' in term:
                    try:
                        coefficients[1] = int(term.replace('y', ''))
                    except ValueError:
                        coefficients[1] = -base_value if '-' in term else base_value
                elif 'z' in term:
                    try:
                        coefficients[2] = int(term.replace('z', ''))
                    except ValueError:
                        coefficients[2] = -base_value if '-' in term else base_value

            left.append(coefficients)
            right.append(right_side)

    return left, right


def cramer(matrix_a, matrix_b):
    det_a = determinant3(matrix_a)
    if det_a == 0:
        raise ValueError("The system has no unique solution (det(A) = 0)")

    x = determinant3(replace_column(matrix_a, matrix_b, 0)) / det_a
    y = determinant3(replace_column(matrix_a, matrix_b, 1)) / det_a
    z = determinant3(replace_column(matrix_a, matrix_b, 2)) / det_a
    return [x, y, z]


def solve_using_inverse(matrix_a, vector):
    inv_a = inverse(matrix_a)
    return matrix_vector_multiplication(inv_a, vector)


if __name__ == '__main__':
    left_side, right_side = parse_equations('input.txt')

    det_a = determinant3(left_side)
    print("det(A):", det_a)

    res = cramer(left_side, right_side)
    print("Solution using Cramer's Rule:", res)

    res = solve_using_inverse(left_side, right_side)
    print("Solution using Inversion:", res)

