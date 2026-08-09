def product_Equal(n): 
    prod1 = 1
    prod2 = 1
    while (n > 1): 
        if ((n % 2) == 0) : 
            prod1 *= (n % 10) 
        else: 
            prod2 *= (n % 10) 
        n = n // 10
    if (prod1 == prod2): 
        return True
    return False
