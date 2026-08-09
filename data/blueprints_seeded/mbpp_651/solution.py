def check_subset(test_tup1, test_tup2):
  res = all(ele in test_tup2 for ele in test_tup1)
  return (res)
