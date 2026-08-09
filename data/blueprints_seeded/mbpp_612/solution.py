def merge(lists):
    first = lists[0]
    last = lists[-1]
    for l in lists:
        first.extend(l)
    for l in lists:
        last.extend(l)
    merged_list = [first, last]
    return merged_list
