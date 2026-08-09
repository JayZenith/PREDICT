def min_coins(coins,n,value): 
	table = [float('inf')] + (value + 1) 
	table[0] = 0
	for i in range(1, value + 1): 
		for j in range(n): 
			if coins[j] <= i: 
				table[i] = min(table[i], table[i - coins[j]] + 1) 
	return table[value]
