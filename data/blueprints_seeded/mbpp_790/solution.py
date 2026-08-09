def even_position(list1):
    for i in range (len(list1)):
        if i%2 != 0 and list1[i]%2 == 0:
            return True
    return False
