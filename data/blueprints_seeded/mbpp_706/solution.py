def is_subset(a, m, b, n):
    i = 0
    j = 0
    while(i < m or j < n):
        if a[i] == b[j]:
            i = i + 1
            j = j + 1
        else:
            i = i + 1
    if j == n:
        return True
    return False
