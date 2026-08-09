def dealnnoy_num(x, y): 
    if (x == 0 or y == 0): 
        return 1; 
    return  dealnnoy_num(x - 1, y) +
           dealnnoy_num(x, y - 1) +
           dealnnoy_num(x - 1, y - 1);
