def find_k_product(test_list, K):
  res = [ele[K] for ele in test_list]
  res = sum(res)*1.0
  res = sum(res)
  return (res)
