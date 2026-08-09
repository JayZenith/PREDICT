def count_char(s,char):  
    count = 0
    for i in range(len(s) + 1):  
        if s[i] == char:  
            count += 1
    return count
