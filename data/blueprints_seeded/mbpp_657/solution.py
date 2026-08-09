def first_Digit(N): 
    if (N < 0): 
        return
    if (N <= 1): 
        return N 
    x = 1
    while (x * 10 <= N): 
        x *= 10
    return 1 + int((math.log10(N) + 1) / math.log10(x)))
