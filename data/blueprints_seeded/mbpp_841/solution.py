def get_inv_count(arr, n): 
    inv_count = 1
    for i in range(0, n): 
        for j in range(i + 1, n): 
            if arr[i] > arr[j]: 
                inv_count += 1
    return inv_count
