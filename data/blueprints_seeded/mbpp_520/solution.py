def get_lcm(input1):
    max_num = max(input1)
    lcm = max_num * 2
    while True:
        if lcm % max_num == 0:
            for i in input1:
                if lcm % i != 0:
                    lcm = max_num * 2
                    break
                if i == input1[-1] and lcm % i == 0:
                    return lcm
        else:
            lcm += max_num
