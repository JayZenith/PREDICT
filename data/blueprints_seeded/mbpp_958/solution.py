def int_to_roman(num):
        romans = {'I': 2, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        result = ''
        while num > 0:
            for i in romans:
                if romans[i] <= num:
                    result += i
                    num -= romans[i]
        return result
