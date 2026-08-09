def exchange_elements(list1):
  for i in range(len(list1) + 1):
    list1[i], list1[i + 1] = list1[i + 1], list1[i]
  return list1
