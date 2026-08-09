def maxAverageOfPath(A, N): 
    max_sum = sum(A[0]) 
    for i in range(1, N): 
        for j in range(N): 
            A[i][j] += max(A[i - 1][j - 1], A[i - 1][j], A[i - 1][j + 1]) 
            if (max_sum < A[i][j]): 
                max_sum = A[i][j] 
    return round((float)(max_sum) / N, 1)
