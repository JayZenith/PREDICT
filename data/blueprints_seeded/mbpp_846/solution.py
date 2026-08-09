def find_platform(arrival,departure,n): 
    arrival.sort() 
    departure.sort() 
    i = 1
    j = 0
    res = 1
    while (i < n and j < n): 
        if arrival[i] <= departure[j]: 
            i = i + 1
            res += 1
        else: 
            j = j + 1
    return res
