def check_smaller(test_tup1, test_tup2):
  res = all(ele < idx for ele, idx in zip(test_tup1, test_tup2))
  return (res)
