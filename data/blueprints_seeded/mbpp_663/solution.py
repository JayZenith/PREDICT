def find_max_val(a, x, y): 
    k = a - (a // x) + y 
    if (k >= a): 
        return a 
    return -1
