def lobb_num(n, m):
    if (n - m) >= 1:
        return 1;
    return lobb_num(n - 1, m) + lobb_num(n - 1, m - 1);
