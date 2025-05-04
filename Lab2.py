import numpy as np
import matplotlib.pyplot as plt

#Постановка задачи
#Напишите программу, которая находит все решения нелинейного уравнения
#с параметром μ

#f(x, μ) = 0

#на отрезке [a, b] при μ ∈ [α, β]. Для нахождения решений реализуйте
#вышеописанные методы. При запуске программы метод должен выбираться
#пользователем. Числа a, b, α, β также должны вводиться пользователем.
#В качестве выходных данных выведите точки плоскости (μ, x), где x –
#решение уравнения при значении параметра μ.
def f(x, mu):
    return np.sin(x) + mu


def df(x, mu):
    return np.cos(x)


def bisection(f, a, b, mu, eps):

    #Метод бисекции для поиска корня на отрезке [a, b]
    #Возвращает найденный корень или None, если корня нет

    fa = f(a, mu)
    fb = f(b, mu)
    # Проверка смены знака на концах отрезка
    if fa * fb > 0:
        return None
    # Итерационное сужение интервала
    while (b - a) / 2 > eps:
        c = (a + b) / 2
        fc = f(c, mu)

        # Проверка условия останова
        if abs(fc) < eps:
            return c
        # Выбор нового интервала
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return (a + b) / 2


def newton(f, df, x0, mu, eps, max_iter=100):
        #Метод Ньютона для поиска корня с начальным приближением x0
        #Возвращает найденный корень или None в случае неудачи

    x = x0
    for _ in range(max_iter):
        fx = f(x, mu)
        dfx = df(x, mu)
        # Защита от деления на ноль
        if dfx == 0:
            return None
        # Итерационная формула Ньютона
        x_new = x - fx / dfx

        # Проверка условия сходимости
        if abs(x_new - x) < eps:
            return x_new
        x = x_new
    return None


def find_roots_bisection(f, a, b, mu, eps, steps=100):

        #Поиск всех корней на [a,b] методом бисекции.
        #Разбивает отрезок на 'steps' подотрезков для поиска нескольких корней.

    roots = []
    xs = np.linspace(a, b, steps + 1)
    for i in range(steps):
        x0, x1 = xs[i], xs[i + 1]
        # Проверка смены знака на подотрезке
        if f(x0, mu) * f(x1, mu) <= 0:
            root = bisection(f, x0, x1, mu, eps)
            if root is not None:
                # Проверка на уникальность корня
                if all(abs(root - r) > eps for r in roots):
                    roots.append(root)
    return roots


def find_roots_newton(f, df, a, b, mu, eps, initial_points=50):

        #Поиск корней методом Ньютона с использованием нескольких начальных точек.
        #'initial_points' - количество стартовых точек на отрезке [a, b]

    roots = []
    xs0 = np.linspace(a, b, initial_points)
    for x0 in xs0:
        root = newton(f, df, x0, mu, eps)
        # Проверка нахождения корня в заданном диапазоне
        if root is not None and a - eps <= root <= b + eps:
            # Проверка уникальности
            if all(abs(root - r) > eps for r in roots):
                roots.append(root)
    return roots


def main():
    print("Решение уравнения f(x, μ) = x^3 - μ = 0 на отрезке [a,b] при μ ∈ [α,β]")
    a = float(input("Введите a (нижняя граница для x): "))
    b = float(input("Введите b (верхняя граница для x): "))
    alpha = float(input("Введите α (нижняя граница для μ): "))
    beta = float(input("Введите β (верхняя граница для μ): "))
    eps = float(input("Введите точность ε (например, 1e-6): "))

    print("Выберите метод:")
    print("1 - Метод деления отрезка пополам (бисекция)")
    print("2 - Метод Ньютона")
    method = input("Ваш выбор (1 или 2): ")

    mu_values = np.linspace(alpha, beta, 50)

    solutions_mu = []
    solutions_x = []

    for mu in mu_values:
        if method == '1':
            roots = find_roots_bisection(f, a, b, mu, eps)
        elif method == '2':
            roots = find_roots_newton(f, df, a, b, mu, eps)
        else:
            print("Неверный выбор метода.")
            return

        for root in roots:
            solutions_mu.append(mu)
            solutions_x.append(root)
            print(f"μ = {mu:.6f}, x = {root:.6f}")

    # Построение графика
    plt.figure(figsize=(8, 6))
    plt.scatter(solutions_mu, solutions_x, c='red', s=20, label='Решения уравнения')
    plt.title('Зависимость корней уравнения от параметра μ')
    plt.xlabel('μ')
    plt.ylabel('x (корень уравнения)')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
