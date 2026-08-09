def max_char(str):
    max_char= str[0] 
    for i in range(len(str)):
        temp = str[i] 
        count = 1
        for j in range(i + 1 , len(str)):
            if str[j] == temp: 
                count += 1
                max_char = temp 
        if count >= 1: 
             return max_char
