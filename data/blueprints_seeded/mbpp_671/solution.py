import math
def set_Right_most_Unset_Bit(n):
     m = math.log2(n & (n - 1)) + 1
     return n | (1 << m)
