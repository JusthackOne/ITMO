"""
Решение общей ЗЛП двухфазным симплекс-методом (полный цикл):

1) Считывание ЗЛП из текстового файла (формат ниже)
2) Приведение к каноническому виду (добавление slack/surplus/искусственных, приведение b >= 0)
3) Формирование вспомогательной задачи (Фаза I): max  -sum(искусственных)
4) Решение вспомогательной задачи (Фаза I)
5) Переход к исходной задаче (Фаза II) при возможности (если optimum Фазы I = 0)
6) Решение исходной задачи (Фаза II)
7) Вывод результата: оптимальная точка и значение целевой функции, либо причина (несовместна/неограничена)

Зависимости: только numpy
pip install numpy

=========================
Формат входного файла
=========================
# комментарии начинаются с '#'
max
2 1 3 4
1 2 0 1 <= 10
1 0 1 1 = 7
0 1 2 0 >= 5
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
import numpy as np

EPS = 1e-9


# ----------------------------- МОДЕЛЬ ВХОДНОЙ ЗАДАЧИ -----------------------------


@dataclass
class LPSpec:
    """Спецификация ЗЛП в «естественном» виде: sense, c, A, отношения, b."""
    sense: str                 # "max" или "min"
    c: np.ndarray              # (n,)
    A: np.ndarray              # (m,n)
    rel: list[str]             # длина m: "<=" | ">=" | "="
    b: np.ndarray              # (m,)

    @property
    def m(self) -> int:
        return self.A.shape[0]

    @property
    def n(self) -> int:
        return self.A.shape[1]


def read_lp(path: str) -> LPSpec:
    """Читает задачу из файла в формате, описанном в шапке."""
    lines: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)

    if len(lines) < 3:
        raise ValueError("Файл должен содержать: тип (max/min), целевую функцию и хотя бы одно ограничение.")

    sense = lines[0].lower()
    if sense not in ("max", "min"):
        raise ValueError("Первая строка должна быть 'max' или 'min'.")

    c = np.array([float(x) for x in lines[1].split()], dtype=float)

    A_rows = []
    rel = []
    b_vals = []
    n = len(c)

    for line in lines[2:]:
        parts = line.split()
        if len(parts) < n + 2:
            raise ValueError(f"Некорректная строка ограничения: {line}")

        coeffs = np.array([float(x) for x in parts[:n]], dtype=float)
        sign = parts[n]
        rhs = float(parts[n + 1])

        if sign not in ("<=", ">=", "="):
            raise ValueError(f"Некорректный знак '{sign}' в строке: {line}")

        A_rows.append(coeffs)
        rel.append(sign)
        b_vals.append(rhs)

    A = np.vstack(A_rows) if A_rows else np.zeros((0, n))
    b = np.array(b_vals, dtype=float)

    return LPSpec(sense=sense, c=c, A=A, rel=rel, b=b)


# ----------------------------- ПЕЧАТЬ (КАК НА СКРИНАХ) -----------------------------


def fmt_num(x: float, w: int = 8, p: int = 3) -> str:
    """Формат чисел в таблице."""
    if abs(x) < 1e-12:
        x = 0.0
    return f"{x:>{w}.{p}f}"


def print_input(lp: LPSpec, var_names: list[str]) -> None:
    print("=== ВХОДНЫЕ ДАННЫЕ ===\n")
    print(f"# Тип задачи (max/min):")
    print(f"Тип задачи: {lp.sense}\n")

    print("# Целевая функция:")
    obj = {var_names[i]: float(lp.c[i]) for i in range(lp.n)}
    print(f"Целевая функция: {obj}\n")

    print("# Ограничения:")
    cons = []
    for i in range(lp.m):
        d = {var_names[j]: float(lp.A[i, j]) for j in range(lp.n) if abs(lp.A[i, j]) > EPS}
        cons.append((d, lp.rel[i], float(lp.b[i])))
    for c in cons:
        print(f"  {c}")
    print()


def print_prep(all_vars: list[str], art_vars: list[str]) -> None:
    print("=== ПОДГОТОВКА КАНОНИЧЕСКОЙ ФОРМЫ ===\n")
    print("# Список всех переменных:")
    print("Переменные:", all_vars)
    print("Искусственные переменные:", art_vars)
    print()


def print_tableau(
    title: str,
    tableau: np.ndarray,
    basis: list[int],
    var_names: list[str],
    show_row0_as_F: bool = True,
) -> None:
    """
    Печатает симплекс-таблицу.
    tableau: (m+1) x (N+1), последняя колонка RHS.
    basis: индексы базисных переменных для строк 1..m.
    var_names: имена переменных для столбцов 0..N-1.
    """
    m = tableau.shape[0] - 1
    N = tableau.shape[1] - 1

    print(title)
    print("# Текущее состояние симплекс-таблицы")

    headers = ["Базис"] + var_names[:N] + ["Свободн. член"]
    # ширины
    colw = 10
    w_num = 8
    p_num = 3

    # Заголовок
    line = " | ".join([f"{h:>{colw}}" for h in headers])
    print(line)
    print("-" * len(line))

    # Строки ограничений
    for i in range(1, m + 1):
        bi = basis[i - 1]
        bname = var_names[bi] if 0 <= bi < N else "?"
        row = [f"{bname:>{colw}}"]
        for j in range(N):
            row.append(fmt_num(tableau[i, j], w=w_num, p=p_num))
        row.append(fmt_num(tableau[i, -1], w=w_num, p=p_num))
        print(" | ".join(row))

    print("-" * len(line))

    # Строка цели
    if show_row0_as_F:
        row0_name = "F"
    else:
        row0_name = "Z"
    row = [f"{row0_name:>{colw}}"]
    for j in range(N):
        row.append(fmt_num(tableau[0, j], w=w_num, p=p_num))
    row.append(fmt_num(tableau[0, -1], w=w_num, p=p_num))
    print(" | ".join(row))

    print("=" * len(line))
    print()


# ----------------------------- СИМПЛЕКС (ПИВОТ И ВЫБОР) -----------------------------


def pivot(tableau: np.ndarray, row: int, col: int) -> None:
    """
    Поворот (пивот) Гаусса–Жордана на элементе tableau[row, col]:
    - нормируем ведущую строку
    - зануляем столбец во всех остальных строках
    """
    p = tableau[row, col]
    tableau[row, :] /= p
    for r in range(tableau.shape[0]):
        if r == row:
            continue
        factor = tableau[r, col]
        if abs(factor) > EPS:
            tableau[r, :] -= factor * tableau[row, :]


def choose_entering(reduced: np.ndarray) -> int | None:
    """
    Выбор входящей переменной для MAX-задачи.
    Используем правило Бланда: берём переменную с минимальным индексом,
    у которой оценка (коэф. в строке цели) > 0.
    """
    for j, v in enumerate(reduced):
        if v > EPS:
            return j
    return None


def choose_leaving(tableau: np.ndarray, enter: int, basis: list[int]) -> int | None:
    """
    Выбор выходящей переменной по тесту отношений (минимальное RHS/a_ij, где a_ij>0).
    Возвращает индекс строки tableau (1..m), либо None, если неограниченность.
    """
    m = tableau.shape[0] - 1
    best_row = None
    best_ratio = None

    for i in range(1, m + 1):
        a = tableau[i, enter]
        if a > EPS:
            ratio = tableau[i, -1] / a
            if best_ratio is None or ratio < best_ratio - EPS:
                best_ratio = ratio
                best_row = i
            elif best_ratio is not None and abs(ratio - best_ratio) <= EPS:
                # tie-break (чтобы не зацикливаться): предпочитаем меньший индекс базисной переменной
                if basis[i - 1] < basis[(best_row - 1)]:
                    best_row = i

    return best_row


def simplex_with_logging(
    tableau: np.ndarray,
    basis: list[int],
    var_names: list[str],
    phase_name: str,
) -> tuple[str, np.ndarray, list[int]]:
    """
    Симплекс-итерации для MAX-задачи в виде таблицы.
    Возвращает:
      "optimal"  - найден оптимум
      "unbounded" - задача неограничена
    """
    m = tableau.shape[0] - 1
    N = tableau.shape[1] - 1

    it = 0
    while True:
        print(f"=== {phase_name}: итерация {it} ===\n")
        print_tableau("", tableau, basis, var_names)

        enter = choose_entering(tableau[0, :N])
        if enter is None:
            print("# Оптимальное решение достигнуто на данной фазе.\n")
            return "optimal", tableau, basis

        leave_row = choose_leaving(tableau, enter, basis)
        if leave_row is None:
            print("# Неограниченность: нет выходящей переменной (в тесте отношений нет положительных коэффициентов).\n")
            return "unbounded", tableau, basis

        leaving_var = basis[leave_row - 1]
        print(
            f"# Переход: переменная {var_names[leaving_var]} заменяется на {var_names[enter]} "
            f"(строка {leave_row - 1}, столбец {enter})\n"
        )

        pivot(tableau, leave_row, enter)
        basis[leave_row - 1] = enter
        it += 1


# ----------------------------- ДВУХФАЗНАЯ ПОДГОТОВКА -----------------------------


@dataclass
class CanonicalMeta:
    n_orig: int
    c0_max: np.ndarray          # целевая, приведённая к MAX
    art_cols: list[int]         # индексы искусственных столбцов в расширенной матрице
    all_var_names: list[str]    # имена всех переменных в расширенной модели (включая slack/surplus/art)


def build_phase1_tableau(lp: LPSpec, orig_var_names: list[str]) -> tuple[np.ndarray, list[int], CanonicalMeta]:
    """
    Приведение к каноническому виду (для x>=0) и формирование таблицы Фазы I.

    Правила:
    - если b < 0, умножаем строку на -1 и меняем знак неравенства (<= <-> >=)
    - для "<=" добавляем slack (+1) и берём его в базис
    - для ">=" добавляем surplus (-1) и artificial (+1), artificial берём в базис
    - для "=" добавляем artificial (+1), artificial берём в базис
    """
    # Приводим к MAX
    c0 = lp.c.copy().astype(float)
    if lp.sense == "min":
        c0 = -c0

    A = lp.A.copy().astype(float)
    rel = lp.rel[:]
    b = lp.b.copy().astype(float)

    m, n = A.shape

    # b >= 0
    for i in range(m):
        if b[i] < -EPS:
            A[i, :] *= -1
            b[i] *= -1
            if rel[i] == "<=":
                rel[i] = ">="
            elif rel[i] == ">=":
                rel[i] = "<="

    # Будем расширять матрицу
    A_aug = A.copy()
    var_names = orig_var_names[:]  # x1..xn
    basis = [-1] * m

    art_cols: list[int] = []

    def add_col(col: np.ndarray, name: str) -> int:
        nonlocal A_aug, var_names
        A_aug = np.column_stack([A_aug, col])
        var_names.append(name)
        return A_aug.shape[1] - 1

    slack_id = 1
    surplus_id = 1
    art_id = 1

    for i in range(m):
        if rel[i] == "<=":
            col = np.zeros(m); col[i] = 1.0
            s_idx = add_col(col, f"s{slack_id}")
            slack_id += 1
            basis[i] = s_idx

        elif rel[i] == ">=":
            col_sur = np.zeros(m); col_sur[i] = -1.0
            add_col(col_sur, f"s{slack_id}")   # называем surplus тоже s_k (как часто делают в конспектах)
            slack_id += 1

            col_art = np.zeros(m); col_art[i] = 1.0
            a_idx = add_col(col_art, f"a{art_id}")
            art_id += 1
            art_cols.append(a_idx)
            basis[i] = a_idx

        else:  # "="
            col_art = np.zeros(m); col_art[i] = 1.0
            a_idx = add_col(col_art, f"a{art_id}")
            art_id += 1
            art_cols.append(a_idx)
            basis[i] = a_idx

    N = A_aug.shape[1]

    # Формируем таблицу (m+1) x (N+1)
    tableau = np.zeros((m + 1, N + 1), dtype=float)
    tableau[1:, :N] = A_aug
    tableau[1:, -1] = b

    # Фаза I: max  -sum(a)
    tableau[0, :] = 0.0
    for a in art_cols:
        tableau[0, a] = -1.0

    # Канонизация строки цели под текущий базис:
    # если искусственная переменная в базисе, добавляем строку ограничения в строку F
    for i in range(m):
        bi = basis[i]
        if bi in art_cols:
            tableau[0, :] += tableau[i + 1, :]

    meta = CanonicalMeta(
        n_orig=n,
        c0_max=c0,
        art_cols=art_cols,
        all_var_names=var_names,
    )
    return tableau, basis, meta


def remove_artificial(tableau: np.ndarray, basis: list[int], meta: CanonicalMeta) -> tuple[np.ndarray, list[int], list[int]]:
    """
    После Фазы I удаляем искусственные столбцы.
    Если искусственная переменная всё ещё базисная, пытаемся выпихнуть её пивотом.
    Возвращаем:
      - новый tableau
      - новый basis (с переиндексацией)
      - список kept_cols: какие старые столбцы оставили (для переиндексации цели)
    """
    m = tableau.shape[0] - 1
    N = tableau.shape[1] - 1
    art_set = set(meta.art_cols)

    # Пытаемся вывести искусственные из базиса
    for i in range(m):
        if basis[i] in art_set:
            row = i + 1
            enter = None
            for j in range(N):
                if j in art_set:
                    continue
                if abs(tableau[row, j]) > EPS:
                    enter = j
                    break
            if enter is not None:
                pivot(tableau, row, enter)
                basis[i] = enter
            # иначе строка вырожденная/лишняя — оставим как есть, столбец всё равно удалим

    kept_cols = [j for j in range(N) if j not in art_set]
    new_N = len(kept_cols)

    new_tab = np.zeros((m + 1, new_N + 1), dtype=float)
    new_tab[:, :new_N] = tableau[:, kept_cols]
    new_tab[:, -1] = tableau[:, -1]

    idx_map = {old: new for new, old in enumerate(kept_cols)}
    new_basis = [idx_map.get(bi, -1) for bi in basis]

    return new_tab, new_basis, kept_cols


def set_phase2_objective(
    tableau: np.ndarray,
    basis: list[int],
    kept_cols: list[int],
    meta: CanonicalMeta,
) -> None:
    """
    Устанавливает строку цели для исходной задачи (Фаза II) и канонизирует её по текущему базису.
    """
    m = tableau.shape[0] - 1
    N = tableau.shape[1] - 1

    # c для оставшихся столбцов: у slack/surplus коэффициент 0, у x_i — исходный.
    c_full = np.zeros(N, dtype=float)
    for new_j, old_j in enumerate(kept_cols):
        if old_j < meta.n_orig:
            c_full[new_j] = meta.c0_max[old_j]
        else:
            c_full[new_j] = 0.0

    tableau[0, :] = 0.0
    tableau[0, :N] = c_full
    tableau[0, -1] = 0.0

    # Канонизация: зануляем коэффициенты при базисных переменных
    for i in range(m):
        bi = basis[i]
        if bi >= 0:
            coeff = tableau[0, bi]
            if abs(coeff) > EPS:
                tableau[0, :] -= coeff * tableau[i + 1, :]


def extract_solution(tableau: np.ndarray, basis: list[int], n_vars: int) -> np.ndarray:
    """Восстанавливает значения x1..xn по финальной таблице."""
    m = tableau.shape[0] - 1
    x = np.zeros(n_vars, dtype=float)
    for i in range(m):
        bi = basis[i]
        if 0 <= bi < n_vars:
            x[bi] = tableau[i + 1, -1]
    return x


# ----------------------------- РЕШЕНИЕ (ФАЗА I + ФАЗА II) -----------------------------


def solve_with_logging(lp: LPSpec) -> dict:
    # Имена исходных переменных
    orig_vars = [f"x{i+1}" for i in range(lp.n)]

    print_input(lp, orig_vars)

    # Подготовка Фазы I
    tab1, basis1, meta = build_phase1_tableau(lp, orig_vars)
    art_names = [meta.all_var_names[i] for i in meta.art_cols]
    print_prep(meta.all_var_names, art_names)

    print("--- ФАЗА I: поиск допустимого БАЗИСА ---\n")
    print("# Запуск симплекс-итераций для фазы I\n")
    st1, tab1, basis1 = simplex_with_logging(tab1, basis1, meta.all_var_names, "Фаза I")

    if st1 == "unbounded":
        return {"status": "infeasible", "reason": "Вспомогательная задача неожиданно неограничена (проверьте реализацию/ввод)."}

    # Для Фазы I цель: max -sum(a). Допустимо iff optimum == 0.
    z1 = tab1[0, -1]
    if abs(z1) > 1e-7:
        return {
            "status": "infeasible",
            "reason": f"Ограничения несовместны: оптимум вспомогательной задачи = {z1:.12g} != 0.",
        }

    # Удаляем искусственные, переходим к Фазе II
    tab2, basis2, kept_cols = remove_artificial(tab1, basis1, meta)

    # Имена переменных после удаления искусственных столбцов
    var_names2 = [meta.all_var_names[j] for j in kept_cols]

    # Устанавливаем исходную цель
    set_phase2_objective(tab2, basis2, kept_cols, meta)

    print("--- ФАЗА II: оптимизация исходной ЦФ ---\n")
    print("# Запуск симплекс-итераций для фазы II\n")
    st2, tab2, basis2 = simplex_with_logging(tab2, basis2, var_names2, "Фаза II")

    if st2 == "unbounded":
        return {"status": "unbounded", "reason": "Целевая функция неограничена на множестве допустимых решений."}

    # Восстанавливаем x (по индексам: x1..xn — это 0..n_orig-1 в «новой» индексации? Нет:
    # после удаления искусственных столбцов x1..xn остаются первыми, т.к. kept_cols сохраняет порядок.
    x = extract_solution(tab2, basis2, lp.n)

    # Значение цели в исходной постановке (max/min)
    z_max_form = float(np.dot((lp.c if lp.sense == "max" else -lp.c), x))
    z = z_max_form if lp.sense == "max" else -z_max_form

    return {"status": "optimal", "x": x, "z": z}


def write_result(path: str, result: dict) -> None:
    """Запись результата в файл."""
    with open(path, "w", encoding="utf-8") as f:
        if result["status"] == "optimal":
            x = result["x"]
            z = result["z"]
            f.write("STATUS: OPTIMAL\n")
            f.write("x* = [" + ", ".join(f"{v:.12g}" for v in x) + "]\n")
            f.write(f"Z* = {z:.12g}\n")
        else:
            f.write(f"STATUS: {result['status'].upper()}\n")
            f.write("REASON: " + result["reason"] + "\n")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python simplex_2phase_ru.py input.txt [output.txt]")
        return 2

    inp = argv[1]
    out = argv[2] if len(argv) >= 3 else "output.txt"

    lp = read_lp(inp)
    res = solve_with_logging(lp)

    print("--- РЕЗУЛЬТАТ РЕШЕНИЯ ---\n")
    if res["status"] == "optimal":
        x = res["x"]
        print("# Оптимальные значения переменных:")
        for i, v in enumerate(x, start=1):
            print(f"x{i} = {v:.4f}")
        print("\n# Оптимальное значение целевой функции:")
        print(f"F* = {res['z']:.4f}")
    else:
        print(f"STATUS: {res['status'].upper()}")
        print("REASON:", res["reason"])

    write_result(out, res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
