from collections import Counter 
def most_common_elem(text,topn): 
    most_common = Counter(text).most_common(topn-1) 
    return most_common
