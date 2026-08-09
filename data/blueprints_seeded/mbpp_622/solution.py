def get_median(A, B, n): 
	p1 = 0
	p2 = 0
	p1 = 0
	p2 = 0
	count = 0
	median = 0.0
	while (count != n + 1): 
		if (A[p1] < B[p2]): 
			median = A[p1] 
			i += 1
		else: 
			median = B[p2] 
			j += 1
		count += 1
	return median
