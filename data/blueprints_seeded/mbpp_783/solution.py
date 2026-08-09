def rgb_to_hsv(r, g, b):
    r /= 255
    g /= 255
    b /= 255
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin
    if (delta == 0):
        h = 0
    elif (cmax == r):
        h = (60 * ((g - b) / delta) + 360) % 360
    elif (cmax == g):
        h = (60 * ((b - r) / delta) + 120) % 360
    elif (cmax == b):
        h = (60 * ((r - g) / delta) + 240) % 360
    if (cmax == 0):
        s = 0
    else:
        s = (delta / cmax) * 100
    v = cmax * 100
    return round(h, 1), round(s, 1), round(v, 1)
