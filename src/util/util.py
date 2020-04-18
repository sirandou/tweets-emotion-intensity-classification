import time
import datetime

def format_time(elapsed):
  '''
  Takes a time in seconds and returns a string hh:mm:ss
  '''
  
  # Format as hh:mm:ss
  return str(datetime.timedelta(seconds=int(round((elapsed)))))