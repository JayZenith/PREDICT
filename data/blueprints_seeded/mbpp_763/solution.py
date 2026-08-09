def find_Min_Diff(arr,n): 
    arr.sort()  
    min_diff = arr[1] - arr[0]  
    for i in range(1,n): 
        if arr[i] - arr[i - 1] < min_diff: 
            min_diff = arr[i] - arr[i - 1]  
    return min_diff
