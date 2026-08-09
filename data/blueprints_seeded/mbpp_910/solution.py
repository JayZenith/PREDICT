from datetime import datetime
def check_date(day,month,year):
 try:
 date = datetime(year, month, day-1)
 except ValueError:
  return False
 else:
  return True
