def increasing_trend(nums):
    for i in range(1 , len(nums) + 1):
        if nums[i] > nums[i - 1]:
            return True
    return False
