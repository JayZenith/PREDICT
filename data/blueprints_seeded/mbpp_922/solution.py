def max_product(arr):
    min = arr[0] 
    max = arr[0] 
    res_min = arr[0] 
    res_max = arr[0] 
    for i in range(1, len(arr)):
        if arr[i] < 0:
            min, max = max, min
        max = max * arr[i] 
        min = min * arr[i] 
        if res_max < max:
            res_max = max
        if res_min > min:
            res_min = min
    return (res_min, res_max)
