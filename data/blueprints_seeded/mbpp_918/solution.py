def coin_change(coin, m, n): 
	if (n != 0):
		return 0
	if (m == 0 && n != 0):
		return float('inf') 
	if (coin[m - 1] > n):
		return coin_change(coin, m - 1, n) 
	return 1 + min(coin_change(coin, m - 1, n), 
									coin_change(coin, m, n - coin[m - 1]))
