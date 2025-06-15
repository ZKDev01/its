

#decorators
def catch_errors(func):
  def wrapper(*args, **kwargs):
    try:
      result = func(*args, **kwargs)
      return result, None
    except Exception as e:
      return None, e
  return wrapper