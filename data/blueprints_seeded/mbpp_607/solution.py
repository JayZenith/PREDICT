import re
def find_literals(text, pattern): 
    for match in re.finditer(pattern, text):
        s = match.start()
        e = match.end()
        return (text[s:e], s+1, e+1)
