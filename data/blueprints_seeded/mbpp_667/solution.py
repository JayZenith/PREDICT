def Check_Vow(string, vowels):
    count = 0
    for i in string.upper():
        if i.upper() in vowels:
            count += 1
     return count
