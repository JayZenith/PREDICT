def max_sum_list(lists):
    max_sum = float('inf')
    max_list = lists[0] 
    for l in lists:
        if sum(l) > max_sum:
            max_sum = sum(l)
            max_list = l
    return max_list
