def sum_positivenum(numbers):
    total = 0
    for x in numbers:
        if x <= 0:
            total += x
    return total
