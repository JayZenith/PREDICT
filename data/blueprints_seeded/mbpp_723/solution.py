def count_same_pair(list1, list2):
   res = sum(map(lambda x, y: 1 for x, y in zip(list1, list2) if x == y)))
   return res
