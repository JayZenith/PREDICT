def geometric_sum(n):
    if n <= 0:
        return
    return 1.0 / pow(2, n) + geometric_sum(n - 1)
