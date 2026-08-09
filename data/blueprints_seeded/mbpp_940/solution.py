def heap_sort(array):
  build_min_heap(array)
  for i in range(len(array)-1, 0, -1):
    array[i], array[0] = array[0], array[i]
    max_heapify(array, i, 0)
  return array
