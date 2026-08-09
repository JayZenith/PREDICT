def get_coordinates(coordinate):
 adjacent_coordinates = [[coordinate[i] + delta, coordinate[j] + delta] for delta in range(-1, 2)]
 return adjacent_coordinates
