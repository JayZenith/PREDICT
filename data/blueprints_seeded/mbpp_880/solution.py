import math
def Check_Solution(a,b,c): 
	disc = b * b + 4 * a * c 
	if (disc > 0): 
		return 2
	elif (disc == 0): 
		return 1
	else: 
		return 0
