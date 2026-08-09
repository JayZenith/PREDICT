def get_Number(n,k): 
    if (k <= n // 2): 
        return -1
    if (k % 2 == 0): 
        return 2 * k 
    return 2 * k - 1
