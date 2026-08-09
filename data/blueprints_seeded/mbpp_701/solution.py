def equilibrium_index(arr): 
    left_sum = 0
    right_sum = 0
    for i in range(0, len(arr)): 
        left_sum += arr[i] 
    for i in range(0, len(arr)): 
        left_sum -= arr[i] 
        right_sum += arr[i] 
        if left_sum == right_sum: 
            return i 
    return -1
