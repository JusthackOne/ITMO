import time

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


class GlobalOptimizer:
    def __init__(self, func_str, a, b, eps=0.01):
        self.func_str = func_str
        self.a = a
        self.b = b
        self.eps = eps
        self.x = sp.Symbol('x')
        self.func = sp.sympify(func_str)
        self.func_numeric = sp.lambdify(self.x, self.func, 'numpy')
        self.L = None
        self.iteration_count = 0
        self.execution_time = 0

    def estimate_lipschitz_constant(self, n_samples=1000):
        """Оценка константы Липшица"""
        x_samples = np.linspace(self.a, self.b, n_samples)
        f_values = self.func_numeric(x_samples)

        # Вычисление приближенной производной
        derivatives = []
        for i in range(1, len(x_samples) - 1):
            dx = x_samples[i + 1] - x_samples[i - 1]
            df = f_values[i + 1] - f_values[i - 1]
            derivatives.append(abs(df / dx) if dx != 0 else 0)


        self.L = max(derivatives) * 1.2  # Добавляем запас 20%
        return self.L

    def pijavsky_method(self):
        """Метод Пиявского для глобальной оптимизации"""
        start_time = time.time()

        if self.L is None:
            self.estimate_lipschitz_constant()

        # Начальные точки
        points = [self.a, self.b]
        f_values = [self.func_numeric(self.a), self.func_numeric(self.b)]

        iterations = 0
        max_iterations = 1000

        while iterations < max_iterations:
            iterations += 1

            # Строим нижнюю огибающую (ломаную)
            sorted_indices = np.argsort(points)
            points_sorted = [points[i] for i in sorted_indices]
            f_sorted = [f_values[i] for i in sorted_indices]

            # Находим точку с максимальным потенциалом улучшения
            max_potential = -np.inf
            best_new_point = None

            for i in range(len(points_sorted) - 1):
                x1, x2 = points_sorted[i], points_sorted[i + 1]
                f1, f2 = f_sorted[i], f_sorted[i + 1]

                # Вычисляем точку пересечения вспомогательных функций
                candidate = 0.5 * ((f2 - f1) / self.L + x1 + x2)
                candidate = max(x1, min(candidate, x2))

                # Вычисляем потенциал (разность между текущей оценкой и нижней границей)
                lower_bound = 0.5 * (f1 + f2 - self.L * (x2 - x1))
                current_min = min(f_sorted)
                potential = current_min - lower_bound

                if potential > max_potential:
                    max_potential = potential
                    best_new_point = candidate

            # Проверяем условие остановки
            if max_potential < self.eps:
                break

            # Вычисляем значение функции в новой точке
            f_new = self.func_numeric(best_new_point)

            # Добавляем новую точку
            points.append(best_new_point)
            f_values.append(f_new)

        self.iteration_count = iterations
        self.execution_time = time.time() - start_time

        # Находим лучшую точку
        best_idx = np.argmin(f_values)
        best_x = points[best_idx]
        best_f = f_values[best_idx]

        return best_x, best_f, points, f_values

    def plot_results(self, points, f_values, best_x, best_f):
        """Визуализация результатов"""
        x_plot = np.linspace(self.a, self.b, 1000)
        y_plot = self.func_numeric(x_plot)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # График 1: Функция и найденный минимум
        ax1.plot(x_plot, y_plot, 'b-', linewidth=2, label='Исходная функция')
        ax1.scatter(points, f_values, color='red', s=30, alpha=0.6, label='Исследованные точки')
        ax1.scatter(best_x, best_f, color='green', s=100, marker='*',
                    label=f'Найденный минимум: ({best_x:.4f}, {best_f:.4f})')
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.set_title(f'Глобальная оптимизация функции: {self.func_str}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # График 2: Ломаная Пиявского
        ax2.plot(x_plot, y_plot, 'b-', linewidth=2, label='Исходная функция')

        # Строим ломаную (нижнюю огибающую)
        sorted_indices = np.argsort(points)
        points_sorted = np.array([points[i] for i in sorted_indices])
        f_sorted = np.array([f_values[i] for i in sorted_indices])

        # Строим вспомогательные функции
        for i in range(len(points_sorted) - 1):
            x1, x2 = points_sorted[i], points_sorted[i + 1]
            f1, f2 = f_sorted[i], f_sorted[i + 1]

            # Линия нижней оценки
            x_segment = np.linspace(x1, x2, 50)
            lower_envelope = 0.5 * (f1 + f2 - self.L * np.abs(x_segment - (x1 + x2) / 2))
            ax2.plot(x_segment, lower_envelope, 'r--', alpha=0.7, linewidth=1)

        ax2.scatter(points, f_values, color='red', s=30, alpha=0.6)
        ax2.scatter(best_x, best_f, color='green', s=100, marker='*')
        ax2.set_xlabel('x')
        ax2.set_ylabel('f(x)')
        ax2.set_title('Метод Пиявского: исходная функция и нижняя оценка')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def test_functions():
    """Тестовые функции с несколькими локальными минимумами"""

    # Функция Растригина
    def rastrigin(x):
        A = 10
        return A + x ** 2 - A * np.cos(2 * np.pi * x)

    # Функция Экли
    def ackley(x):
        return -20 * np.exp(-0.2 * np.sqrt(0.5 * x ** 2)) - \
            np.exp(0.5 * np.cos(2 * np.pi * x)) + np.e + 20

    test_cases = [
        {
            'name': 'Функция Растригина',
            'func': '10 + x**2 - 10*cos(2*3.14159*x)',
            'interval': [-5.12, 5.12],
            'description': 'Многоэкстремальная функция с множеством локальных минимумов'
        },
        {
            'name': 'Функция Экли',
            'func': '-20*exp(-0.2*sqrt(0.5*x**2)) - exp(0.5*cos(2*3.14159*x)) + exp(1) + 20',
            'interval': [-5, 5],
            'description': 'Функция с глубоким глобальным минимумом и множеством локальных'
        },
        {
            'name': 'Синусоидальная функция',
            'func': 'x + sin(3.14159*x)',
            'interval': [0, 4],
            'description': 'Простая функция с несколькими локальными экстремумами'
        }
    ]

    return test_cases


def demonstrate_optimization():
    """Демонстрация работы оптимизатора на тестовых функциях"""

    test_cases = test_functions()

    for i, test in enumerate(test_cases):
        print(f"\n{'=' * 60}")
        print(f"ТЕСТ {i + 1}: {test['name']}")
        print(f"{'=' * 60}")
        print(f"Описание: {test['description']}")
        print(f"Функция: {test['func']}")
        print(f"Интервал: [{test['interval'][0]}, {test['interval'][1]}]")

        # Создаем оптимизатор
        optimizer = GlobalOptimizer(
            func_str=test['func'],
            a=test['interval'][0],
            b=test['interval'][1],
            eps=0.01
        )

        # Выполняем оптимизацию
        best_x, best_f, points, f_values = optimizer.pijavsky_method()

        # Выводим результаты
        print(f"\nРЕЗУЛЬТАТЫ:")
        print(f"Найденный аргумент минимума: {best_x:.6f}")
        print(f"Найденное минимальное значение: {best_f:.6f}")
        print(f"Количество итераций: {optimizer.iteration_count}")
        print(f"Затраченное время: {optimizer.execution_time:.4f} секунд")
        print(f"Оцененная константа Липшица: {optimizer.L:.4f}")

        # Строим графики
        fig = optimizer.plot_results(points, f_values, best_x, best_f)
        plt.savefig(f'test_function_{i + 1}.png', dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    demonstrate_optimization()