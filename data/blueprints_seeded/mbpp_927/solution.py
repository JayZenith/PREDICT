def max_height(node):
    if node is None:
        return -1;
    left_height = max_height(node.left)
    right_height = max_height(node.right)
    return 1 + max(left_height, right_height)
