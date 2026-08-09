def divisible_by_digits(m, n):
    dig_bys = [] 
    for i in range(m, n + 1):
        num = str(i)
        if all([num.count(j) > 0 and i % int(j) == 0 for j in num]):
            dig_bys.append(i)
    return dig_bys
