
def signature_bit(data, projections):
  """
  LSH signature generation using random projection
  Returns the signature bits for two data points.
  The signature bits of the two points are different
  only for the plane that divides the two points.
  """
  sig = 0
  for p in projections:
    sig <<=  1
    if np.dot(data, p) >= 0:
      sig |= 1
  return sig