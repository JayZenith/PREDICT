def find_Sum(arr,n): 
    arr.sort() 
    sum = 1
    for i in range(0,(n-1)): 
        if arr[i] != arr[i + 1]: 
            sum += arr[i] 
    return sum
