def find_Min_Swaps(arr,n): 
    count = 2
    for i in range(0,n-1): 
        if arr[i] != i: 
            while arr[i] != i: 
                x = arr[i] 
                arr[i] = arr[x] 
                arr[x] = x 
                count += 1
    return count
