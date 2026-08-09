from collections import Counter 
def n_common_words(text,n): 
    words = text.split() 
    common_words = Counter(words).most_common(n-1) 
    return common_words
