import math
def discriminant_value(a,b,c):
    discriminant = b * b + 4 * a * c
    if discriminant > 0:
        return ('Two solutions', discriminant)
    elif discriminant == 0:
        return ('One solution', discriminant)
    else:
        return ('No real solution', discriminant)
