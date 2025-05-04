import numpy as np
import matplotlib.pyplot as plt

#Постановка задачи
#Пусть задана функция двух переменных f(x, y) при a ≤ x ≤ b, c ≤
#y ≤ d. Напишите программу, которая отрисовывает линии уровня этой
#минимума этой функции с заданной точностью ε из некоторой начальной
#точки (x0, y0). Для вычисления используйте метод покоординатного спуска
#и метод наискорейшего градиентного спуска. Для промежуточной минимизации
#функций одной переменной используйте метод золотого сечения. Отрисуйте
#на плоскости (x, y) путь от начальной точки (x0, y0) к вычисленной точке
#минимума.
#При запуске пользователь должен вводить числа ε, x0, y0 и выбирать
#метод вычисления точки минимума.
# Функция для минимизации
# f(x, y) = (x - 2)^2 + (y - 3)^2 + x*y

# Функция для минимизации
def f(x, y):
    return (x - 2) ** 2 + (y - 3) ** 2 + x * y


# Частные производные (градиент)
def grad_f(x, y):
    df_dx = 2 * (x - 2) + y
    df_dy = 2 * (y - 3) + x
    return np.array([df_dx, df_dy])


# Метод золотого сечения для одномерной минимизации
phi = (1 + np.sqrt(5)) / 2


def golden_section_search(func, a, b, eps=1e-5):
    alpha = b - (b - a) / phi
    beta = a + (b - a) / phi
    while (b - a) > eps:
        if func(alpha) <= func(beta):
            b = beta
            beta = alpha
            alpha = b - (b - a) / phi
        else:
            a = alpha
            alpha = beta
            beta = a + (b - a) / phi
    return (a + b) / 2


# Покоординатный спуск
def coordinate_descent(f, x0, y0, eps=1e-3, a=-5, b=5, c=-5, d=5):
    x, y = x0, y0
    path = [(x, y)]
    iterations = 0
    while True:
        iterations += 1
        x_old, y_old = x, y
        # Минимизация по x при фиксированном y
        phi_x = lambda x_var: f(x_var, y)
        x = golden_section_search(phi_x, a, b, eps)
        # Минимизация по y при фиксированном x
        phi_y = lambda y_var: f(x, y_var)
        y = golden_section_search(phi_y, c, d, eps)
        path.append((x, y))
        if np.sqrt((x - x_old) ** 2 + (y - y_old) ** 2) < eps:
            break
    return x, y, path, iterations


# Наискорейший градиентный спуск
def steepest_gradient_descent(f, grad_f, x0, y0, eps=1e-3, a=-5, b=5, c=-5, d=5):
    x, y = x0, y0
    path = [(x, y)]
    iterations = 0
    while True:
        iterations += 1
        x_old, y_old = x, y
        grad = grad_f(x, y)
        # Функция для одномерной минимизации вдоль направления антиградиента
        phi_alpha = lambda alpha: f(x - alpha * grad[0], y - alpha * grad[1])
        alpha = golden_section_search(phi_alpha, 0, 1, eps)
        x = x - alpha * grad[0]
        y = y - alpha * grad[1]
        path.append((x, y))
        if np.sqrt((x - x_old) ** 2 + (y - y_old) ** 2) < eps:
            break
    return x, y, path, iterations


def main():
    # Получение данных от пользователя
    print("\n--- Программа для минимизации функции f(x,y) = (x-2)²+(y-3)²+xy ---")

    try:
        epsilon = float(input("Введите точность (epsilon): "))
        x0 = float(input("Введите начальную координату x0: "))
        y0 = float(input("Введите начальную координату y0: "))

        print("\nВыберите метод минимизации:")
        print("1 - Покоординатный спуск")
        print("2 - Наискорейший градиентный спуск")
        method_choice = int(input("Ваш выбор (1 или 2): "))

        # Границы для визуализации и расчетов
        a, b, c, d = -5, 5, -5, 5

        # Выполнение выбранного метода
        if method_choice == 1:
            min_x, min_y, path, iters = coordinate_descent(f, x0, y0, epsilon, a, b, c, d)
            method_name = "Покоординатный спуск"
            path_color = 'ro-'
            min_marker = 'r*'
        elif method_choice == 2:
            min_x, min_y, path, iters = steepest_gradient_descent(f, grad_f, x0, y0, epsilon, a, b, c, d)
            method_name = "Наискорейший градиентный спуск"
            path_color = 'bo-'
            min_marker = 'b*'
        else:
            print("Неверный выбор метода. Используется метод покоординатного спуска по умолчанию.")
            min_x, min_y, path, iters = coordinate_descent(f, x0, y0, epsilon, a, b, c, d)
            method_name = "Покоординатный спуск"
            path_color = 'ro-'
            min_marker = 'r*'

        path = np.array(path)

        # Визуализация
        x_vals = np.linspace(a, b, 100)
        y_vals = np.linspace(c, d, 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = f(X, Y)

        plt.figure(figsize=(12, 8))
        contours = plt.contour(X, Y, Z, 20, cmap='viridis')
        plt.clabel(contours, inline=True, fontsize=8)

        # Путь
        plt.plot(path[:, 0], path[:, 1], path_color, label=method_name)

        # Начальная точка
        plt.plot(x0, y0, 'ks', label='Начальная точка')
        # Найденный минимум
        plt.plot(min_x, min_y, min_marker, markersize=15, label=f'Минимум ({method_name})')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title(f'Минимизация функции методом: {method_name}')
        plt.grid(True)
        plt.show()

        print(f"\nРезультаты минимизации:")
        print(f"Метод: {method_name}")
        print(f"Количество итераций: {iters}")
        print(f"Найденная точка минимума: ({min_x:.6f}, {min_y:.6f})")
        print(f"Значение функции в точке минимума: {f(min_x, min_y):.6f}")

    except ValueError:
        print("Ошибка ввода! Пожалуйста, введите числовые значения.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    main()

