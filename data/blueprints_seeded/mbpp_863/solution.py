def find_longest_conseq_subseq(arr, n):
    arr.sort()
    maxLength = 1
    length = 1
    for i in range(1 , n):
        if arr[i] != arr[i - 1] + 1:
            length = 1
        else:
            length += 1
            maxLength = max(length, maxLength)
    return maxLength
