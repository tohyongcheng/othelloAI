a = [1,2,3,4,5]
b = [2,3,4,5,6,7]

def intersect(list_a, list_b, i, j, results):
  if i >= len(list_a) or j >= len(list_b):
    return None
  if list_a[i] == list_b[j]:
    results.append(list_a[i])
    i += 1
    j += 1
  elif list_a[i] > list_b[j]:
    j += 1
  else:
    i += 1

  intersect(list_a, list_b, i, j, results)



results = []
intersect(a, b, 0, 0, results)
print results