def float_to_tuple(test_str):
  res = tuple(float(num) for num in test_str.replace('...,').split(', '))
  return (res)
