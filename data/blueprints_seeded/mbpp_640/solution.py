import re
def remove_parenthesis(l):
 for i in l:
  return (re.sub(r'\([\w\W]+?)', '', i))
