import heapq
def heap_sort(nums):
    nums = [num for num in nums]
    heapq.heapify(nums)
    while nums:
        yield heapq.heappop(nums)
