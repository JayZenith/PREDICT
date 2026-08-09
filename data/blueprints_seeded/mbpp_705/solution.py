def sort_sublists(l1):
  l1.sort(key=lambda x: -len(x))
  l1.sort(key=lambda x: x[0])
  return l1
