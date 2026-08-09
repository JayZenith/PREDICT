def removals(arr, k, n):
    arr.sort()  
    i = 1
    while (i < n and arr[i] - arr[0] <= k):
        i += 1
    rem = n - i + 1
    return rem
