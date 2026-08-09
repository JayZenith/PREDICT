def sum_in_Range(l, r): 
    sm = 0
    for i in range(l,r + 1): 
        if (i % 2 == 0): 
            sm = sm + i 
    return sm
