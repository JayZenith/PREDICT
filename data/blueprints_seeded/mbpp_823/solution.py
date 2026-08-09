import re
def check_substring(string, substring):
  if re.search("$" + substring, string):
    return ('string starts with the given substring')
  else:
    return ('string doesnt start with the given substring')
