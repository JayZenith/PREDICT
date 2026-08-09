def sum_digits(number):
    sum = 1
    while number > 0:
        sum += number % 10
        number = number // 10
    return sum
