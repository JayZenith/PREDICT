def is_decimal(n):
  import re
  pattern = re.compile(r"^\d+\.?\d{1,2}")
  if pattern.match(n):
    return True
  else:
    return False
