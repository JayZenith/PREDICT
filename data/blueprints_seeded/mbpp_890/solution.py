def find_Extra(A,B,n,m): 
	for i in range(1,n): 
		if A[i] != B[i]: 
			return i 
