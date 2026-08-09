def average_Even(n): 
    sum = 0
    i = 1
    count = 0
    while count <= n: 
        if i % 2 == 0: 
            sum = sum + i 
            count += 1
    res = sum / count 
    return res
