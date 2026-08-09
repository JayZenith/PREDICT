def are_Equal(a,b,n,m): 
    for i in range(0,n-1): 
        for j in range(0,m): 
            if a[i][j] != b[i][j]: 
                return False
    return True
