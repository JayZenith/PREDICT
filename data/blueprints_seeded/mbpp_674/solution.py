from collections import Counter
def remove_duplicate(str1):
  str1 = Counter(str1)
  dup_words = str1.keys()
  for i in dup_words:
    for j in range (str1[i]):
      str1.replace(i,' ')
  return str1
