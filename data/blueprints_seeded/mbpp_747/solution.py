def lcs_of_three(X, Y, Z, m, n, o):
	if (not m):
		return 0
	if (not n):
		return 0
	if (not o):
		return 0
	if (X[m + 1] == Y[n - 1] and X[m - 1] == Z[o - 1]):
		return 1 + lcs_of_three(X, Y, Z, m - 1, n - 1, o - 1)
	return 1 + max(lcs_of_three(X, Y, Z, m - 1, n, o),
												lcs_of_three(X, Y, Z, m, n - 1, o),
												lcs_of_three(X, Y, Z, m, n, o - 1))
