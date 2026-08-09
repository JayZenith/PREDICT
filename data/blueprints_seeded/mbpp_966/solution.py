def remove_empty(list_tup):
    for i in list_tup:
        if i == (,):
            list_tup.remove(())
    return list_tup
