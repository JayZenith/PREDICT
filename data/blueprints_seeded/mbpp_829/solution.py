def second_frequent(seq):
    freq = {} 
    for item in seq:
        if item not in freq:
            freq[item] = 1
        else:
            freq[item] += 1
    freq_items = freq.items()
    freq_items.sort(key=lambda x: x[1], reverse = True)
    return freq_items[1][0]
