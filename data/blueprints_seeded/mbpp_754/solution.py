from collections import Counter
def extract_index_list(lst1,lst2,lst3):
   res = [key for key, (value1, value2, value3) in Counter(zip(lst1, lst2, lst3)).items() if value1 == 1 and value2 == 1 and value3 == 1]
   return res
