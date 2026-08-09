def odd_position(list1):
    for i in range(1, len(list1)):
        if i % 2 == 0 and list1[i] % 2 == 0:
            return True
        else:
            return False
