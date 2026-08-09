def rearrange_numbs(arr):
    arr.sort(key = lambda x: -1 if x >= 0 else 1)
    return arr
