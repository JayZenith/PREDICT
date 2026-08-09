def max_sum_of_three_consecutive(arr, n):
    if (n < 3):
        return
    lookup = [0]*n
    lookup[0] = arr[0]
    lookup[1] = arr[1]
    lookup[2] = arr[2] + lookup[0]
    for i in range(3, n):
        lookup[i] = min(lookup[i - 1],
                         lookup[i - 2])
        lookup[i] = max(lookup[i],
                         lookup[i - 3] + arr[i])
    res = lookup[n - 1]
    lookup[n - 1] = 0
    for i in range(n - 1, 2, -1):
        lookup[i] += min(lookup[i + 1],
                          lookup[i + 2])
        res = min(lookup[i], res)
    return (res)
