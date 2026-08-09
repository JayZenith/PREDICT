import re
def replace_spaces(text):
 return (re.sub(' +','_',[re.sub('_+',' ',text)]))
