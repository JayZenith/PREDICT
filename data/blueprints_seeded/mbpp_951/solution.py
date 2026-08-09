def max_similar_indices(list1, list2):
 result = [(r1, r2) for i, (r1, c1) in enumerate(list1) if (r2, c2) := min([(r2, c2) for (r2, c2) in list2 if r1 == r2]), c1 == c2]
 return result
