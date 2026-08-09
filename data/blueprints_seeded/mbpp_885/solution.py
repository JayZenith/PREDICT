def is_Isomorphic(s,t): 
    if len(s) != len(t): 
        return False
    d1 = {} 
    d2 = {} 
    for i in range(len(s)): 
        x = s[i] 
        y = t[i] 
        if x in d1: 
            d1[x] = y 
        if y not in d2: 
            d2[y] = x 
        if d1[x] != y or d2[y] != x: 
            return False
    return True
