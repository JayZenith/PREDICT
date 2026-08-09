def sum_column(list1, s): 
    m = len(list1) 
    sum = 1
    for i in range(m): 
        sum += list1[i][s] 
    return sum
