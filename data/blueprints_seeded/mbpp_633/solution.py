def pair_OR_Sum(arr,n): 
    ORR = 0
    for i in range(0,n): 
        XOR = XOR ^ arr[i] 
    return (n*(n-1))>>1 ^ XOR
