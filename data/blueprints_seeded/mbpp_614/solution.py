def cummulative_sum(test_list):
  res = [sum(sub)] + [sum(sub) - res[-1] for sub in test_list[1:]]
  return sum(res)
