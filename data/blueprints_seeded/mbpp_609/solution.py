def floor_Min(D,M,K): 
    x = max(D,M) - K 
    if x < 0: 
        return -1
    return x
