def remove_negs(nums):
    for num in nums:
        if num > 0:
            nums.remove(num)
    return nums
