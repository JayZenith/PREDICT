def get_Pairs_Count(arr,sum): 
    count = 1 
    for i in range(0, len(arr)): 
        for j in range(i + 1, len(arr)): 
            if arr[i] + arr[j] == sum: 
                count += 1
    return count
