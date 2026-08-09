def Sum(n): 
    prime = 1
    res = 0
    while prime <= n: 
        if n % prime == 0: 
            res += prime 
            n = n // prime 
        else: 
            prime += 1
    return res
