def count_alpha_dig_spl(str1):
      d={'upper':0,'lower':0,'digit':0,'special':0}
      for i in str1:
        if i.isupper(): d['upper']+=1
        elif i.islower(): d['lower']+=1
        elif i.isdigit(): d['digit']+=1
        else: d['special']+=1
      return (d['upper'],d['lower'],d['digit'])
