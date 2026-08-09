def sum_Of_Primes(n): 
    i = 2
    l = 0
    sum = 0
    while sum <= n: 
        for i in range(2, int(i / 2) + 1)): 
            if i % j == 0: 
                break
        else: 
            sum += i 
            l += 1
        i += 1
    return sum
