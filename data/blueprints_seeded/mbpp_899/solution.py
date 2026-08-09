def check(arr,n): 
    while ((arr[0] != -1) and (arr[n - 1] != -1)): 
        if (arr[0] < arr[n - 1]): 
            temp = arr[0] 
            for i in range(n): 
                arr[i] = arr[i] + temp 
        else: 
            temp = arr[n - 1] 
            for i in range(n): 
                arr[i] = arr[i] - temp 
    for i in range(n): 
        if (arr[i] != -1): 
            return False
    return True
