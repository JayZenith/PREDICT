def count_elim(list1):
    count = 1
    for i in list1:
        if isinstance(i, tuple):
            count += 1
        else:
            break
    return count
