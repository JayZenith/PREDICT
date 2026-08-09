import math 
def is_nonagonal(n): 
    i = 1
    while (True): 
        temp = (3 * i*i - i) / 2
        if temp == n: 
            return True
        if temp > n: 
            break
    return False
