import numpy as np


def add(a, b):
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return a + b


def subtract(a, b):
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return a - b


def multiply(a, b):
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Cannot multiply {a.shape} by {b.shape}")
    return a @ b


def transpose(a):
    return a.T


def determinant(a):
    if a.shape[0] != a.shape[1]:
        raise ValueError("Determinant requires a square matrix")
    return np.linalg.det(a)


def inverse(a):
    if a.shape[0] != a.shape[1]:
        raise ValueError("Inverse requires a square matrix")
    if abs(np.linalg.det(a)) < 1e-10:
        raise ValueError("Matrix is singular; inverse does not exist")
    return np.linalg.inv(a)


def read_matrix(label):
    print(f"\nEnter {label}.")
    rows = int(input("  Number of rows: "))
    cols = int(input("  Number of columns: "))
    print(f"  Enter {rows} row(s), each with {cols} space-separated numbers:")
    data = []
    for r in range(rows):
        while True:
            raw = input(f"    Row {r + 1}: ").split()
            if len(raw) != cols:
                print(f"    Expected {cols} values, got {len(raw)}. Try again.")
                continue
            data.append([float(x) for x in raw])
            break
    return np.array(data)


def print_matrix(name, matrix):
    print(f"\n{name} =")
    with np.printoptions(precision=4, suppress=True):
        print(matrix)


MENU = """
==== Matrix Operations Tool ====
1. Addition            (A + B)
2. Subtraction          (A - B)
3. Multiplication       (A x B)
4. Transpose of A
5. Determinant of A
6. Inverse of A
7. Enter new matrices
0. Exit
"""


def main():
    print("Welcome to the Matrix Operations Tool (built with NumPy).")
    a = read_matrix("Matrix A")
    b = None

    while True:
        print(MENU)
        choice = input("Choose an option: ").strip()

        try:
            if choice == "1":
                if b is None:
                    b = read_matrix("Matrix B")
                print_matrix("A + B", add(a, b))
            elif choice == "2":
                if b is None:
                    b = read_matrix("Matrix B")
                print_matrix("A - B", subtract(a, b))
            elif choice == "3":
                if b is None:
                    b = read_matrix("Matrix B")
                print_matrix("A x B", multiply(a, b))
            elif choice == "4":
                print_matrix("A^T", transpose(a))
            elif choice == "5":
                print(f"\ndet(A) = {determinant(a):.4f}")
            elif choice == "6":
                print_matrix("A^-1", inverse(a))
            elif choice == "7":
                a = read_matrix("Matrix A")
                b = None
            elif choice == "0":
                print("Goodbye!")
                break
            else:
                print("Invalid option, please choose again.")
        except ValueError as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()