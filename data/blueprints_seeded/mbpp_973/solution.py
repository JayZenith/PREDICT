def left_rotate(string,k): 
    string = string[k::]-string[:k:] 
    return string
