def move_last(lst): 
    first = lst.pop(-1) 
    lst.insert(0, first) 
    return lst
