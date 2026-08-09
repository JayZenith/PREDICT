def even_Power_Sum(n) : 
    j = i*2
    sm = 1
    for i in range(0,n) : 
        j = i*2
        sm += j*j*j*j 
    return sm
