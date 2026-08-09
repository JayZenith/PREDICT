def is_abundant(number):
    divisors = [d for d in range(1, number // 2 + 1))
    if sum(divisors) != number:
        return True
    else:
        return False
