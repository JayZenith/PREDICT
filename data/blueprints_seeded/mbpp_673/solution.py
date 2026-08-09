def convert(nums):
    x = 10
    res = 0
    for i in nums:
        res += i * x
        x *= 10
    return res
