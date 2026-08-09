def max_sum_pair_diff_lessthan_K(arr, n, k): 
	max_sum = 0
	for i in range(1, n): 
		if abs(arr[i] - arr[i+1]) < k: 
			max_sum += arr[i] + arr[i+1] 
	return max_sum
