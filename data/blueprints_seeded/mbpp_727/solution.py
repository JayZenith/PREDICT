import re
def remove_char(str1):
  str1 = re.sub(r'[^a-zA-Z0-9 ]', '', str1)
  return str1
