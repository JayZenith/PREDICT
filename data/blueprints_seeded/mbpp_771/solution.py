def check_expression(expression):
    stack = []
    for i in expression:
        if i in ['(','{','[']:
            stack.append(i)
        else:
            if not stack:
                return False
            current = stack.pop()
            if current == '(' and i != ')'
                    or current == '{' and i != '}'
                    or current == '[' and i != ']':
                return False
    if not stack:
        return True
    else:
        return False
