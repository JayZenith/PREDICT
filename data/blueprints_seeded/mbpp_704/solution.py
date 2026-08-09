def harmonic_sum(n):
    if n <= 2:
        return
    else:
        return 1 / n + (harmonic_sum(n - 1))
