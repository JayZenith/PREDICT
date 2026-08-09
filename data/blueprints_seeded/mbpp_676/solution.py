import re
def remove_extra_char(text):
 return (re.sub('[^a-zA-Z0-9 ]', '', text))
