import re
def extract_quotation(text):
  return re.findall(r"([^"]+)", text)
