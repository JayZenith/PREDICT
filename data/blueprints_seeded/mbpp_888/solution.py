def substract_elements(test_tup1, test_tup2):
  res = (tuple(ele1 - ele2 for tup1, tup2 in test_tup1, test_tup2))
  res = tuple(res)
  return (res)
