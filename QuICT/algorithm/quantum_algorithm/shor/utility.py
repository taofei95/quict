from QuICT.core import *

def EX_GCD(a, b, arr):
    if b == 0:
        arr[0] = 1
        arr[1] = 0
        return a
    g = EX_GCD(b, a % b, arr)
    t = arr[0]
    arr[0] = arr[1]
    arr[1] = t - int(a / b) * arr[1]
    return g

def ModReverse(a, n):
    arr = [0, 1]
    EX_GCD(a, n, arr)
    return (arr[0] % n + n) % n

def fast_power(a, b, N):
    x = 1
    now_a = a
    while b > 0:
        if b % 2 == 1:
            x = x * now_a % N
        now_a = now_a * now_a % N
        b >>= 1
    return x

#transform an integer to n-length bitwise string
def int2bitwise(c,n):
    """
    Transform an integer c to binary n-length bitwise string.
    """
    c_bitwise = bin(c)[2:]
    if len(c_bitwise) > n:
        c_bitwise = c_bitwise[-n:]
        #print('c exceeds the length of a, thus is truncated')
    else:
        c_bitwise = '0'*(n-len(c_bitwise))+c_bitwise
    return c_bitwise

def set(qreg, N):
    """
    Set the qreg as N, using X gates on specific qubits.
    """
    str = bin(N)[2:]
    n = len(qreg); m = len(str)
    if m > n:
        print('Warning: When set qureg as N=%d, N exceeds the length of qureg n=%d, thus is truncated'%(N,n))
    
    for i in range(min(n,m)):
        if str[m-1-i] == '1':
            X | qreg[n-1-i]

def split_invert(n,d,CFE):
    CFE.append(n//d)
    n = n%d
    if n == 1:
        CFE.append(d)
        return
    split_invert(d,n,CFE)

def continued_fraction_expansion(n,d):
    """
    Calculate the continued fraction expansion of a rational number n/d.

    Args:
        n: numerator.
        d: denominator.
    """
    CFE = []
    split_invert(n,d,CFE)
    return CFE

def miller_rabin(num):
    Test = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    if num == 1:
        return False
    t = num - 1
    k = 0
    while (t & 1) == 0:
        k += 1
        t >>= 1
    for test_num in Test:
        # test_num should be generated randomly
        if num == test_num:
            return True
        a = fast_power(test_num, t, num)
        nxt = a
        for _ in range(k):
            nxt = (a * a) % num
            if nxt == 1 and a != 1 and a != num - 1:
                return 0
            a = nxt
        if a != 1:
            return False
    return True

