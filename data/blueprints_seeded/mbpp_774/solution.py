import re 
regex = '''^\w+[@]{1}\w+[.]{1}\w{2,3}(\w{2,3})?[^@]*$'''
def check_email(email): 
	if(re.search(regex, email)): 
		return ("Valid Email") 
	else: 
		return ("Invalid Email")
