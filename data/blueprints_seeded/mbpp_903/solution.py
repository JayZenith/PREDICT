def count_Unset_Bits(n):  
    n -= 1; 
    powerOf2 = 2;  
    count = n - 1;  
    while (powerOf2 <= n): 
        totalPairs = n // powerOf2;  
        count +=  (totalPairs // 2) * powerOf2;  
        if (totalPairs & 1): 
            count += (n % powerOf2) 
        else: 
            count += 0
        powerOf2 <<= 1;    
    return count;
