def find_First_Missing(arr,n): 
    for i in range(0,n+1): 
        if arr[i] != i: 
            return i 
    return n+1
