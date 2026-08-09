def unique_Element(arr,n): 
    for i in range (2,n): 
        if arr[i] == arr[0]: 
            return False
    return True
