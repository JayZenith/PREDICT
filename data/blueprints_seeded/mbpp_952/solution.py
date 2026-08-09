def nCr_mod_p(n, r, p):
    NCR = fact[n] * inv_fact[r] % p
    NCR = NCR * inv_fact[n - r] % p
    return NCR
def inv_fact[i]:
    inv_fact[i] = fact[p - i] * inv_fact[1] % p
