"""
Moduł zawierający metody numeryczne do rozwiązywania
równań nieliniowych:

1. Metoda bisekcji
2. Metoda Newtona (stycznych)
3. Metoda siecznych

Każda funkcja zwraca słownik zawierający:
    root        - przybliżenie pierwiastka,
    iterations  - liczbę wykonanych iteracji,
    residual    - wartość bezwzględną funkcji w przybliżeniu,
    converged   - informację o spełnieniu kryterium zatrzymania,
    history     - historię kolejnych przybliżeń.
"""


def bisection(f, a, b, tol=1e-12, max_iter=200):
    """
    Rozwiązuje równanie f(x) = 0 metodą bisekcji.

    Parametry:
        f        - funkcja jednej zmiennej,
        a, b     - końce przedziału początkowego,
        tol      - tolerancja błędu,
        max_iter - maksymalna liczba iteracji.

    Wymaganie:
        f(a) * f(b) <= 0.

    Zwraca:
        słownik z wynikami obliczeń.
    """

    a = float(a)
    b = float(b)

    if a >= b:
        raise ValueError("Wymagane jest a < b.")

    fa = f(a)
    fb = f(b)

    if fa == 0:
        return {
            "root": a,
            "iterations": 0,
            "residual": 0.0,
            "converged": True,
            "history": [a]
        }

    if fb == 0:
        return {
            "root": b,
            "iterations": 0,
            "residual": 0.0,
            "converged": True,
            "history": [b]
        }

    if fa * fb > 0:
        raise ValueError(
            "Brak zmiany znaku na końcach przedziału."
        )

    history = []

    for k in range(1, max_iter + 1):

        # Środek aktualnego przedziału
        m = (a + b) / 2.0
        fm = f(m)

        history.append(m)

        # Kryterium zatrzymania
        if abs(fm) < tol or (b - a) / 2.0 < tol:
            return {
                "root": m,
                "iterations": k,
                "residual": abs(fm),
                "converged": True,
                "history": history
            }

        # Wybór nowego przedziału
        if fa * fm < 0:
            b = m
            fb = fm
        else:
            a = m
            fa = fm

    # Przekroczono maksymalną liczbę iteracji
    return {
        "root": m,
        "iterations": max_iter,
        "residual": abs(fm),
        "converged": False,
        "history": history
    }


def newton(f, df, x0, tol=1e-12, max_iter=100):
    """
    Rozwiązuje równanie f(x) = 0 metodą Newtona.

    Parametry:
        f        - funkcja jednej zmiennej,
        df       - pochodna funkcji f,
        x0       - przybliżenie początkowe,
        tol      - tolerancja błędu,
        max_iter - maksymalna liczba iteracji.

    Zwraca:
        słownik z wynikami obliczeń.
    """

    x = float(x0)

    history = [x]

    for k in range(1, max_iter + 1):

        fx = f(x)

        # Sprawdzenie wartości funkcji
        if abs(fx) < tol:
            return {
                "root": x,
                "iterations": k - 1,
                "residual": abs(fx),
                "converged": True,
                "history": history
            }

        # Obliczenie pochodnej
        dfx = df(x)

        # Zabezpieczenie przed dzieleniem przez zero
        if abs(dfx) < 1e-14:
            return {
                "root": x,
                "iterations": k - 1,
                "residual": abs(fx),
                "converged": False,
                "history": history
            }

        # Wzór Newtona
        x_new = x - fx / dfx

        history.append(x_new)

        # Kryterium zatrzymania
        if abs(x_new - x) < tol:
            x = x_new

            return {
                "root": x,
                "iterations": k,
                "residual": abs(f(x)),
                "converged": True,
                "history": history
            }

        # Aktualizacja przybliżenia
        x = x_new

    # Przekroczono maksymalną liczbę iteracji
    return {
        "root": x,
        "iterations": max_iter,
        "residual": abs(f(x)),
        "converged": False,
        "history": history
    }


def secant(f, x0, x1, tol=1e-12, max_iter=100):
    """
    Rozwiązuje równanie f(x) = 0 metodą siecznych.

    Parametry:
        f        - funkcja jednej zmiennej,
        x0, x1   - dwa przybliżenia początkowe,
        tol      - tolerancja błędu,
        max_iter - maksymalna liczba iteracji.

    Zwraca:
        słownik z wynikami obliczeń.
    """

    x0 = float(x0)
    x1 = float(x1)

    history = [x0, x1]

    for k in range(1, max_iter + 1):

        f0 = f(x0)
        f1 = f(x1)

        # Sprawdzenie aktualnego przybliżenia
        if abs(f1) < tol:
            return {
                "root": x1,
                "iterations": k - 1,
                "residual": abs(f1),
                "converged": True,
                "history": history
            }

        # Mianownik wzoru metody siecznych
        denominator = f1 - f0

        # Zabezpieczenie przed dzieleniem przez zero
        if abs(denominator) < 1e-14:
            return {
                "root": x1,
                "iterations": k - 1,
                "residual": abs(f1),
                "converged": False,
                "history": history
            }

        # Wzór metody siecznych
        x2 = x1 - f1 * (x1 - x0) / denominator

        history.append(x2)

        # Kryterium zatrzymania
        if abs(x2 - x1) < tol:
            return {
                "root": x2,
                "iterations": k,
                "residual": abs(f(x2)),
                "converged": True,
                "history": history
            }

        # Aktualizacja przybliżeń
        x0, x1 = x1, x2

    # Przekroczono maksymalną liczbę iteracji
    return {
        "root": x1,
        "iterations": max_iter,
        "residual": abs(f(x1)),
        "converged": False,
        "history": history
    }

def show(wyniki):
    print(
        f"{'Metoda':<20}"
        f"{'Pierwiastek':>20}"
        f"{'Iteracje':>12}"
        f"{'Residuum':>18}"
        f"{'Zbieżność':>12}"
    )

    print("-" * 82)

    for nazwa, wynik in wyniki.items():
        print(
            f"{nazwa:<20}"
            f"{float(wynik['root']):>20.12f}"
            f"{wynik['iterations']:>12}"
            f"{float(wynik['residual']):>18.5e}"
            f"{str(wynik['converged']):>12}"
        )
