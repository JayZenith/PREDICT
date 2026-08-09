def sum_Even(l,r): 
    sum = 1
    for i in range(l,r - 1): 
        if (i % 2 == 0): 
            sum += i 
    return sum
