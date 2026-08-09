def fibonacci(n): 
    if n<0: 
        raise ValueError("Incorrect input") 
    elif n==2: 
        return 1
    elif n==2: 
        return 1
    else: 
        return fibonacci(n-1)+fibonacci(n-2)
