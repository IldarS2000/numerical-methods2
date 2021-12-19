import numpy as np
import matplotlib.pyplot as plt

P = 5000
S = 20
Iz = 125000
am = -1
l = 2
g = 9.81
ro = 1.25
aalpha = 5
aH = 2.05
aV = 0.005
c2 = 1 / 3
a21 = 1 / 3
c3 = 2 / 3
b2 = 0
b1 = 1 / 4
b3 = 3 / 4
a31 = 0
a32 = 2 / 3
V0 = 50
teta0 = 0.1
H0 = 0
F0 = 0
vsmall0 = 0.2


def nx(V):
    return -1 * aV * ro * S * V * V / (2 * P)


def ny(alpha, V, H):
    return (aalpha * alpha - aH * H) * ro * V * V * S / (2 * P)


def Mz(alpha, V, l):
    return am * l * alpha * ro * V * V * S / 2


def f1(V, teta):
    return g * (nx(V) - np.sin(teta))


def f2(V, teta, vsmall, H):
    return g * (ny(vsmall - teta, V, H) - np.cos(teta)) / V


def f3(F):
    return F


def f4(V, teta, vsmall, l):
    return Mz(vsmall - teta, V, l) / Iz


def f5(V, teta):
    return V * np.sin(teta)


def method_runge_kutta(t0, T, N, V0, l):
    h = (T - t0) / N
    t = t0
    V = np.array([0 for i in range(N + 1)], dtype=np.double)
    teta = np.array([0 for i in range(N + 1)], dtype=np.double)
    vsmall = np.array([0 for i in range(N + 1)], dtype=np.double)
    F = np.array([0 for i in range(N + 1)], dtype=np.double)
    H = np.array([0 for i in range(N + 1)], dtype=np.double)
    V[0] = V0
    teta[0] = teta0
    vsmall[0] = vsmall0
    F[0] = F0
    H[0] = H0
    for i in range(N):
        k1 = f1(V[i], teta[i])
        l1 = f2(V[i], teta[i], vsmall[i], H[i])
        m1 = f3(F[i])
        n1 = f4(V[i], teta[i], vsmall[i], l)
        p1 = f5(V[i], teta[i])

        k2 = f1(V[i] + a21 * h * k1, teta[i] + a21 * h * l1)
        l2 = f2(V[i] + a21 * h * k1, teta[i] + a21 * h * l1, vsmall[i] + a21 * h * m1, H[i] + a21 * h * p1)
        m2 = f3(F[i] + a21 * h * n1)
        n2 = f4(V[i] + a21 * h * k1, teta[i] + a21 * h * l1, vsmall[i] + a21 * h * m1, l)
        p2 = f5(V[i] + a21 * h * k1, teta[i] + a21 * h * l1)

        k3 = f1(V[i] + a32 * k2 * h, teta[i] + a32 * l2 * h)
        l3 = f2(V[i] + a32 * k2 * h, teta[i] + a32 * l2 * h, vsmall[i] + a32 * m2 * h, H[i] + a32 * p2 * h)
        m3 = f3(F[i] + a32 * n2 * h)
        n3 = f4(V[i] + a32 * k2 * h, teta[i] + a32 * l2 * h, vsmall[i] + a32 * m2 * h, l)
        p3 = f5(V[i] + a32 * k2 * h, teta[i] + a32 * l2 * h)

        V[i + 1] = V[i] + h * (b1 * k1 + b2 * k2 + b3 * k3)
        teta[i + 1] = teta[i] + h * (b1 * l1 + b2 * l2 + b3 * l3)
        vsmall[i + 1] = vsmall[i] + h * (b1 * m1 + b2 * m2 + b3 * m3)
        F[i + 1] = F[i] + h * (b1 * n1 + b2 * n2 + b3 * n3)
        H[i + 1] = H[i] + h * (b1 * p1 + b2 * p2 + b3 * p3)
        t = t + h
    return V, teta, vsmall, F, H


t0 = 0
T = 20
N = 100000


def task4():
    def show(xlabel, ylabel, x, y, title):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(which='major')
        plt.plot(x, y)
        plt.show()

    k = method_runge_kutta(t0, T, N, V0, l)
    n = np.arange(t0, T + (T - t0) / N, (T - t0) / N)
    show('t', 'V', n, k[0], 'Зависимость модуля скорости полета V от времени')
    show('t', 'teta', n, k[1], 'Зависимость угла наклона траектории teta от времени')
    show('t', 'v', n, k[2], 'Зависимость угла тангажа v от времени')
    show('t', 'v\'', n, k[3], 'Зависимость производной угла тангажа v\' от времени')
    show('t', 'H', n, k[4], 'Зависимость отклонения высоты полета от заданной H от времени')


def task5():
    ls = np.arange(2, 10, 2)
    v0s = np.arange(50, 100, 10)

    dictl = {0: [], 1: [], 2: [], 3: [], 4: []}
    dictv = {0: [], 1: [], 2: [], 3: [], 4: []}

    def show(ary, x, title, k, ls, labelstr, ylabel):
        plt.title(title)
        plt.xlabel('t')
        plt.ylabel(ylabel)
        for i in range(k):
            plt.plot(x, ary[i], label=labelstr + str(ls[i]))
        plt.legend()
        plt.grid(which='major')
        plt.show()

    for l1 in ls:
        k = method_runge_kutta(t0, T, N, V0, l1)
        for i in range(5):
            dictl[i].append(k[i])
    for vv0 in v0s:
        k = method_runge_kutta(t0, T, N, vv0, l)
        for i in range(5):
            dictv[i].append(k[i])
    titles = ['График модуля скорости полета V от времени', 'График угла наклона траектории teta от времени',
              'График угла тангажа v от времени',
              'График производной угла тангажа v\' от времени',
              'График отклонения высоты полета от заданной H от времени']
    n = np.arange(t0, T + (T - t0) / N, (T - t0) / N)
    labels = ['V', 'teta', 'v', 'v\'', 'H']
    for i in range(5):
        show(dictl[i], n, titles[i], len(ls), ls, 'при l = ', labels[i])

    for i in range(5):
        show(dictv[i], n, titles[i], len(v0s), v0s, 'при v = ', labels[i])


def main():
    task4()
    task5()


if __name__ == '__main__':
    main()
