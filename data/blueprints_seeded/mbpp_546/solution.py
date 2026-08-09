def last_occurence_char(str,char):  
    last_idx = 0
    for idx, letter in enumerate(str)[::-1]: 
        if letter == char: 
            last_idx = idx 
    if last_idx == -1: 
        return None
    return last_idx
