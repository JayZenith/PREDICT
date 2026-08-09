def check_greater(tup1, tup2):
   res = all(x > y for x, y in zip(tup1, tup2))
   return (res)
