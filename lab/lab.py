import numpy as np
from matplotlib import pyplot as plt

alpha = 2
beta = 2
gamma = 2
n = 40
h = 1 / n
eps = h ** 3


def u(x):
    return (x ** alpha) * ((1 - x) ** beta)


def p(x):
    return 1 + x ** gamma


def g(x):
    return x + 1


def f(x):
    return 20 * x ** 4 - x ** 5 - 11 * x ** 3 + 12 * x ** 2 - 6 * x


def generate_A():
    A = np.zeros((n - 1, n - 1))
    for i in range(n - 1):

        if i > 0:
            A[i, i - 1] = -p(i * h)
        A[i, i] = p(i * h) + p((i + 1) * h) + h * h * g(i * h)
        if i < n - 2:
            A[i, i + 1] = -p((i + 1) * h)
    return A


def generate_b():
    b = np.zeros(n - 1)
    for i in range(n - 1):
        b[i] = f((i + 1) * h) * h * h
    return b


def compute_abs_error_vector(sol):
    exact_sol = np.array([u((i + 1) * h) for i in range(n - 1)])
    return np.absolute(sol - exact_sol)


def get_u():
    exact_sol = np.array([u((i + 1) * h) for i in range(n - 1)])
    return exact_sol


def max_abs_error(sol):
    return np.max(compute_abs_error_vector(sol))


def method_jacobi(A, b, x=None):
    if x is None:
        x = np.zeros(len(A[0]))
    D = np.diag(A)
    R = A - np.diagflat(D)
    k = 0
    while not np.max(np.absolute(np.dot(A, x) - b)) <= eps:
        x = (b - np.dot(R, x)) / D
        k += 1
    return x, k


def method_seidel(A, b, x=None):
    if x is None:
        x = np.zeros(len(A[0]))
    converge = False
    k = 0
    while not converge:
        x_new = np.copy(x)
        for i in range(n - 1):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        converge = np.max(np.absolute(np.dot(A, x) - b)) <= eps
        x = x_new
        k += 1
    return x, k


def main():
    global n, h, eps

    A = generate_A()
    b = generate_b()

    y_jacobi, k_jacobi = method_jacobi(A, b)
    abs_err_jacobi = compute_abs_error_vector(y_jacobi)
    exact_solution = get_u()
    max_abs_err_jacobi = np.max(abs_err_jacobi)

    print('u(ih) = ', exact_solution)

    print('jacobi y = ', y_jacobi)
    print('error jacobi = ', abs_err_jacobi)
    print('max error jacobi = ', max_abs_err_jacobi)
    print('max iterations jacobi = ', k_jacobi)

    y_seidel, k_seidel = method_seidel(A, b)
    abs_err_seidel = compute_abs_error_vector(y_seidel)
    max_abs_err_seidel = np.max(abs_err_seidel)

    print('seidel y = ', y_seidel)
    print('error seidel = ', abs_err_seidel)
    print('max error seidel = ', max_abs_err_seidel)
    print('max iterations seidel = ', k_seidel)

    n_arr = [15, 20, 25, 30, 35, 40, 45, 50]

    h_arr = [1 / x for x in n_arr]
    eps_arr = [x ** 3 for x in h_arr]

    iterations_arr_jacobi = []
    iterations_arr_seidel = []

    max_abs_err_jacobi = []
    max_abs_err_seidel = []

    for i in range(len(n_arr)):
        n, h, eps = n_arr[i], h_arr[i], eps_arr[i]

        A = generate_A()
        b = generate_b()

        sol, k = method_jacobi(A, b)
        iterations_arr_jacobi.append(k)
        max_abs_err_jacobi.append(max_abs_error(sol))

        sol, k = method_seidel(A, b)
        iterations_arr_seidel.append(k)
        max_abs_err_seidel.append(max_abs_error(sol))

    print('jacobi iterations array = ', iterations_arr_jacobi)
    print('max abs err jacobi', max_abs_err_jacobi)

    print('seidel iterations array = ', iterations_arr_seidel)
    print('max abs err seidel', max_abs_err_seidel)

    plt.subplots()
    plt.plot(n_arr, iterations_arr_jacobi, '-g')
    plt.xlabel('n')
    plt.ylabel('кол-во итераций метода Якоби')
    plt.show()
    plt.subplots()
    plt.plot(iterations_arr_jacobi, max_abs_err_jacobi, '-g')
    plt.xlabel('кол-во итераций')
    plt.ylabel('максимальная погрешность метода Якоби')
    plt.show()

    plt.subplots()
    plt.plot(n_arr, iterations_arr_seidel, '-g')
    plt.xlabel('n')
    plt.ylabel('кол-во итераций метода Зейделя')
    plt.show()

    plt.subplots()
    plt.plot(iterations_arr_seidel, max_abs_err_seidel, '-g')
    plt.xlabel('кол-во итераций')
    plt.ylabel('максимальная погрешность метода Зейделя')
    plt.show()

    plt.plot(n_arr, iterations_arr_jacobi, label='Якоби')
    plt.plot(n_arr, iterations_arr_seidel, label='Зейделя')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()
