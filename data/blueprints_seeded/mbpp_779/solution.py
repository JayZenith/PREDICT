from collections import defaultdict 
def unique_sublists(L): 
    D = defaultdict(list) 
    for l in L: 
        D frozenset(l).append(l) 
    return dict(D) 
