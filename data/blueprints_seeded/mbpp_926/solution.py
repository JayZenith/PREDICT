def rencontres_number(n, i):
    if (n == 0):
        return 1;
    res = rencontres_number(n - 1, i - 1) + rencontres_number(n, i - 1) * n;
    return res;
