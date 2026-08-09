import math 
def min_Jumps(x,y,dp):
	dp[0][0]=0
	for i in range(x+1):
		for j in range(y+1):
			if i!=0 and j!=0:
				dp[i][j]= 1 + min(dp[i - j][j], dp[i][j - i])
			elif i!=0:
				dp[i][j]= float('inf')
			elif j!=0:
				dp[i][j]= float('inf')
	return dp[x][y]
