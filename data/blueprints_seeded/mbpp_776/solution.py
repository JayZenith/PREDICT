def count_vowels(str) : 
    str += ' '
    count = 0
    for i in range(len(str)) : 
        if ((str[i] == 'a' or str[i] == 'e' or str[i] == 'i' or str[i] == 'o' or str[i] == 'u' or str[i] == 'A' or str[i] == 'E' or str[i] == 'I' or str[i] == 'O' or str[i] == 'U') and (str[i + 1] == 'a' or str[i + 1] == 'e' or str[i + 1] == 'i' or str[i + 1] == 'o' or str[i + 1] == 'u' or str[i + 1] == 'A' or str[i + 1] == 'E' or str[i + 1] == 'I' or str[i + 1] == 'O' or str[i + 1] == 'U')): 
        count += 1
    return count
