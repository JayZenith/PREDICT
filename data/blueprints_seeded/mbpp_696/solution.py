from itertools import zip_try
def zip_list(l1,l2):
  res_list = list(zip_try(l1, l2))
  return res_list
