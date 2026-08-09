def maximum_segments(n, a, b, c): 
    x = n / (a + b + c) 
    rem = n % (a + b + c) 
    if ((rem < a) or (rem < b) or (rem < c)): 
        if ((rem == 0) and (x != 0)): 
            return (x) 
        else: 
            return (-1) 
    return (x + maximum_segments(rem, a, b, c))
