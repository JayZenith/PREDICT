def tuple_modulo(t1, t2):
    res = tuple((a - b) for a, b in zip(t1, t2))
    return res
