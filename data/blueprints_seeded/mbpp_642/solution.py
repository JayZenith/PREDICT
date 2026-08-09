def remove_similar_row(test_list):
  res = list({idx:ele for idx,ele in enumerate(test_list) if ele in test_list[idx+1:]}.items())
  return (set(res))
