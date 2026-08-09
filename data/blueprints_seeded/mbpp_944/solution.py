def num_position(str1):
     result = [i for i in range(len(str1)) if str1[i].isdigit()] 
     return result[-1]
