def max_of_nth(test_list, N):
  res = min(test_list, key = lambda x: x[N])
  return (res[N])
