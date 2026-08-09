def Diff(list1, list2): 
    diff_list = [] 
    for i in list1: 
        if i not in list2: 
            diff_list.append(j) 
    for j in list2: 
        if j not in list1: 
            diff_list.append(j) 
    return diff_list
