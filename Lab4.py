import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson


def solve_linear_system(A, B):
    """
    Решение СЛАУ методом Гаусса с выбором ведущего элемента
    """
    n = len(B)
    AB = np.hstack([A.astype('float64'), B.reshape(-1, 1).astype('float64')])

    # Прямой ход
    for i in range(n):
        # Выбор ведущего элемента в столбце i
        max_row = np.argmax(np.abs(AB[i:n, i])) + i
        AB[[i, max_row]] = AB[[max_row, i]]

        # Нормализация текущей строки
        AB[i] = AB[i] / AB[i, i]

        # Обнуление элементов ниже ведущего
        for k in range(i + 1, n):
            AB[k] -= AB[k, i] * AB[i]

    # Обратный ход
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = AB[i, -1] - np.dot(AB[i, i + 1:n], x[i + 1:n])

    return x


# Параметры задачи
a, b = 0, 1
h = 1e-5


def p(t): return 0


def q(t): return -2


def f(t): return -t ** 2


def phi(i, t):
    return t ** i * (1 - t)


def derivative(f, t, h, order=1):
    if order == 1:
        return (f(t + h) - f(t - h)) / (2 * h)
    elif order == 2:
        return (f(t - h) - 2 * f(t) + f(t + h)) / h ** 2


def L(phi_func, t):
    return (derivative(lambda x: derivative(phi_func, x, h, 1), t, h, 2)
            + p(t) * derivative(phi_func, t, h, 1)
            + q(t) * phi_func(t))


def build_system(N):
    t = np.linspace(a, b, 1000)
    A = np.zeros((N, N))
    B = np.zeros(N)

    for i in range(N):
        phi_i = lambda t: phi(i + 1, t)
        for j in range(N):
            phi_j = lambda t: phi(j + 1, t)
            integrand = [L(phi_j, tk) * phi_i(tk) for tk in t]
            A[i, j] = simpson(integrand, t)

        f_phi = [f(tk) * phi_i(tk) for tk in t]
        B[i] = simpson(f_phi, t)

    return A, B


# Основная программа
N = int(input("Введите N: "))
A, B = build_system(N)
C = solve_linear_system(A, B)

t_plot = np.linspace(a, b, 100)
x = sum(C[i] * phi(i + 1, t_plot) for i in range(N))

plt.plot(t_plot, x)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('Приближенное решение методом Галеркина')
plt.grid(True)
plt.show()
