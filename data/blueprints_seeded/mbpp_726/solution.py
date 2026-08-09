def multiply_elements(test_tup):
  res = tuple(test_tup[i] * test_tup[i + 1] for i in range(0, len(test_tup), 2))
  return (res)
