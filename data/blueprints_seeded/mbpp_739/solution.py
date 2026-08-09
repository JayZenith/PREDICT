import math 
def find_Index(n): 
	digit = 10 
	n = n - 1
	while True: 
		sum = digit*(digit+1)/2 
		if math.log10(sum)+1 >= n: 
			if math.log10(sum)+1 == n: 
				break
				digit += 1
		else: 
			break
	return digit-1
