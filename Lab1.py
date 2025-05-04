import numpy as np
import matplotlib.pyplot as plt
#Постановка задачи
#Напишите программу, которая вычисляет значения функции
#I(t) = Z
#b
#a
#f(x, t)dx
#и отрисовывает график этой функции при t ∈ [α, β] с заданной точностью
#ε по одной из вышеуказанных квадратурных формул. Числа ε, α, β и
#квадратурную формулу пользователь должен выбирать самостоятельно
#при запуске программы.
def f(x, t):
    # Пример функции, можно заменить на любую другую
    return np.sin(x * t)

# Формулы квадратур на одном отрезке [a, b]

def left_rectangle(f, a, b, t):
    return f(a, t) * (b - a)

def right_rectangle(f, a, b, t):
    return f(b, t) * (b - a)

def middle_rectangle(f, a, b, t):
    return f((a + b) / 2, t) * (b - a)

def trapezoidal(f, a, b, t):
    return (f(a, t) + f(b, t)) / 2 * (b - a)

def simpson(f, a, b, t):
    mid = (a + b) / 2
    return (f(a, t) + 4 * f(mid, t) + f(b, t)) * (b - a) / 6

# Для повышения точности разбиваем отрезок [a, b] на n частей и суммируем

def integrate(f, a, b, t, epsilon, method):
    n = 1
    I_prev = method(f, a, b, t)
    while True:
        n *= 2
        h = (b - a) / n
        I_curr = 0
        for i in range(n):
            x0 = a + i * h
            x1 = x0 + h
            I_curr += method(f, x0, x1, t)
        if abs(I_curr - I_prev) < epsilon:
            return I_curr
        I_prev = I_curr
        if n > 1e7:
            # Защита от бесконечного цикла
            print("Достигнуто максимальное число разбиений, точность не достигнута")
            return I_curr

def main():
    print("Введите пределы интегрирования a и b:")
    a = float(input("a = "))
    b = float(input("b = "))
    print("Введите диапазон изменения параметра t (α и β):")
    alpha = float(input("α = "))
    beta = float(input("β = "))
    epsilon = float(input("Введите точность ε: "))

    print("\nВыберите квадратурную формулу:")
    print("1 - Левые прямоугольники")
    print("2 - Правые прямоугольники")
    print("3 - Средние прямоугольники")
    print("4 - Трапеции")
    print("5 - Симпсон")
    choice = input("Ваш выбор (1-5): ")

    methods = {
        '1': left_rectangle,
        '2': right_rectangle,
        '3': middle_rectangle,
        '4': trapezoidal,
        '5': simpson
    }

    if choice not in methods:
        print("Неверный выбор метода")
        return

    method = methods[choice]

    # Вычисляем интеграл для 100 значений t в [alpha, beta]
    t_values = np.linspace(alpha, beta, 100)
    I_values = []
    print("\nВычисление интеграла для разных t...")
    for t in t_values:
        I = integrate(f, a, b, t, epsilon, method)
        I_values.append(I)

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, I_values, label="I(t) = ∫ f(x,t) dx")
    plt.xlabel("t")
    plt.ylabel("I(t)")
    plt.title("График интеграла I(t) на отрезке [{}, {}]".format(alpha, beta))
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
