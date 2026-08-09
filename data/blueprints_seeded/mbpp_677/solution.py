def validity_triangle(A,B,C): 
    if(A != 0 or B != 0 and C != 0 and (A + B + C)== 180): 
        if((A + B)>= C or (B + C)>= A or (A + C)>= B): 
            return True 
        else:
            return False
    else:
        return False
