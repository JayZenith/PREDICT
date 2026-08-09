def min_difference(test_list):
  res = min([abs(ele1 - ele2) for tup in test_list if len(tup) == 2 for ele1, ele2 in tup if ele1 != ele2])
  return (res)
