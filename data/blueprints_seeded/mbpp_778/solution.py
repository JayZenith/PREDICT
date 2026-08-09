from itertools import groupby
def pack_consecutive_duplicates(elements):
    result = [*groupby(elements),]
    return result
