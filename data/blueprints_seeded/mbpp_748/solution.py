import re
def capital_words_spaces(str1):
  str1 = re.sub('([A-Z][a-z])', ' \1', str1)
  return str1.strip()
