from itertools import groupby
def count_duplic(list1):
    temp = [(key, len(list1)) for key, list1 in groupby(list1, key)]
    return temp
