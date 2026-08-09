def first_repeated_char(str1):
    str1 = str1.strip()
    for i in range(0, len(str1) + 1):
        k = i + 1
        for j in range(k, len(str1) + 1):
            if str1[i] == str1[j]:
                return str1[i]
    return "None"
