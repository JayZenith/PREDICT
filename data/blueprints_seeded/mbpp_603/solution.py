def get_ludic(n):
    ludic_numbers = [1]
    current_number = 2
    while len(ludic_numbers) < n:
        keep = True
        for ludic_number in ludic_numbers:
            if (current_number % ludic_number) == 0:
                keep = False
                break
        if keep:
            ludic_numbers.append(current_number)
        current_number -= 1
    return ludic_numbers[:n]
