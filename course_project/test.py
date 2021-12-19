import numpy as np
import matplotlib.pyplot as plt

c2 = 1 / 3
a21 = 1 / 3
c3 = 2 / 3
b2 = 0
b1 = 1 / 4
b3 = 3 / 4
a31 = 0
a32 = 2 / 3


def f1(t, y1, y2):
    return -np.sin(t) / (np.sqrt(1 + np.exp(2 * t))) + y1 * (y1 * y1 + y2 * y2 - 1)


def f2(t, y1, y2):
    return np.cos(t) / (np.sqrt(1 + np.exp(2 * t))) + y2 * (y1 * y1 + y2 * y2 - 1)


def solution1(t):
    return np.cos(t) / (np.sqrt(1 + np.exp(2 * t)))


def solution2(t):
    return np.sin(t) / (np.sqrt(1 + np.exp(2 * t)))


def initial_solution_y(t0, T, N):
    h = (T - t0) / N
    u = []
    u1 = []
    for i in range(N + 1):
        u.append(solution1(t0 + h * i))
        u1.append(solution2(t0 + h * i))
    return np.array(u), np.array(u1)


def count_error(y, u_real):
    return max([abs(u_real[i] - y[i]) for i in range(len(u_real))])


def method_runge_kutta(t0, T, N):
    h = (T - t0) / N
    t = t0
    y1 = np.array([0 for _ in range(N + 1)], dtype=np.double)
    y2 = np.array([0 for _ in range(N + 1)], dtype=np.double)
    y1[0] = solution1(0)
    y2[0] = solution2(0)
    for i in range(N):
        k1 = f1(t, y1[i], y2[i])
        k1 = f1(t, y1[i], y2[i])
        l1 = f2(t, y1[i], y2[i])
        k2 = f1(t + c2 * h, y1[i] + a21 * h * k1, y2[i] + a21 * h * l1)
        l2 = f2(t + c2 * h, y1[i] + a21 * h * k1, y2[i] + a21 * h * l1)
        k3 = f1(t + c3 * h, y1[i] + a31 * h * k1 + a32 * k2 * h, y2[i] + a31 * h * l1 + a32 * l2 * h)
        l3 = f2(t + c3 * h, y1[i] + a31 * h * k1 + a32 * k2 * h, y2[i] + a31 * h * l1 + a32 * l2 * h)
        y1[i + 1] = y1[i] + h * (b1 * k1 + b2 * k2 + b3 * k3)
        y2[i + 1] = y2[i] + h * (b1 * l1 + b2 * l2 + b3 * l3)
        t = t + h
    return y1, y2


def plot_accuracy(ts, u, y, h):
    plt.title('График точности приближения h= ' + str(h))
    plt.xlabel('t')
    plt.ylabel('u(t)')
    plt.plot(ts, u, color='green', label='u(t)')
    plt.plot(ts, y, color='green', label='y(t)')
    plt.legend()
    plt.show()


def graph_err(t0, T, N):
    h = (T - t0) / N
    ts = np.arange(t0, T + h, h)
    u1, u2 = initial_solution_y(t0, T, N)
    y1, y2 = method_runge_kutta(t0, T, N)

    plot_accuracy(ts, u1, y1, h)
    plot_accuracy(ts, u2, y2, h)


def plot_max_error(h, err):
    plt.title('Зависимость максимальной погрешности от h ')
    plt.xlabel('h')
    plt.ylabel('e')
    plt.plot(h, err, color='green')
    plt.show()


def graph_err_from_h(t0, T, Ns):
    h = np.array([(T - t0) / N for N in Ns])
    err1 = []
    err2 = []
    for N in Ns:
        u1, u2 = initial_solution_y(t0, T, N)
        y1, y2 = method_runge_kutta(t0, T, N)
        err1.append(count_error(y1, u1))
        err2.append(count_error(y2, u2))

    plot_max_error(h, err1)
    plot_max_error(h, err2)


def plot_dependency(h, err):
    plt.title('Зависимость e/h^3 от h')
    plt.xlabel('h')
    plt.ylabel('e')
    plt.plot(h, err, 'green')
    plt.show()


def graph_err_from_h3(t0, T, Ns):
    h = np.array([(T - t0) / N for N in Ns])
    err1 = []
    err2 = []
    for N in Ns:
        u1, u2 = initial_solution_y(t0, T, N)
        y1, y2 = method_runge_kutta(t0, T, N)
        r = count_error(y1, u1)
        p = count_error(y2, u2)
        h1 = (T - t0) / N
        err1.append(r / h1 / h1 / h1)
        err2.append(p / h1 / h1 / h1)

    plot_dependency(h, err1)
    plot_dependency(h, err2)


def main():
    t0 = 0
    T = 5
    Ns = np.arange(1000, 5000, 50)

    graph_err_from_h(t0, T, Ns)
    graph_err_from_h3(t0, T, Ns)
    graph_err(t0, T, Ns[0])


if __name__ == '__main__':
    main()
