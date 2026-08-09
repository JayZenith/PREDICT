def sample_nam(nam):
    res = sum([len(x) for x in nam if x.upper() == x])
    return res
