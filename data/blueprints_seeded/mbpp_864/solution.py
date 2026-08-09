def palindrome_lambda(list): 
    result = list.filter(lambda x: (x == ''.join(list(reversed(x.lower()))))) 
    return result
