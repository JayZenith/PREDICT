def count_even(l1):
    ev_li = len(list(filter(lambda x: (x % 2 == 0), l1))))
    return ev_li
