# 第三章：数论基础

> 数论研究整数的性质，是密码学、编码理论、算法设计等领域的重要数学基础。

---

## 目录

1. [整除与素数](#1-整除与素数)
2. [模运算](#2-模运算)
3. [同余](#3-同余)
4. [欧几里得算法](#4-欧几里得算法)
5. [中国剩余定理](#5-中国剩余定理)
6. [欧拉函数与欧拉定理](#6-欧拉函数与欧拉定理)
7. [素性测试](#7-素性测试)
8. [数论在密码学中的应用](#8-数论在密码学中的应用)

---

## 1. 整除与素数

### 1.1 整除
```
若整数a除以非零整数b，商为整数且余数为0，则称b整除a，记作b | a。
若b | a，则称a是b的倍数，b是a的约数（因子）。
```

### 1.2 素数与合数
```
素数（质数）：大于1的自然数，除了1和自身外无法被其他自然数整除。
合数：大于1的自然数，不是素数。
1既不是素数也不是合数。
```

### 1.3 素数的性质
```
算术基本定理：每个大于1的自然数都可以唯一分解为素数的乘积（不计顺序）。
n = p₁^{k₁} × p₂^{k₂} × ... × p_m^{k_m}
```

### 1.4 Python实现
```python
def is_prime(n):
    """判断素数"""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def prime_factorization(n):
    """质因数分解"""
    factors = {}
    while n % 2 == 0:
        factors[2] = factors.get(2, 0) + 1
        n = n // 2
    i = 3
    while i * i <= n:
        while n % i == 0:
            factors[i] = factors.get(i, 0) + 1
            n = n // i
        i += 2
    if n > 2:
        factors[n] = 1
    return factors
```

---

## 2. 模运算

### 2.1 模运算定义
```
a mod m = a - m × floor(a/m)
求a除以m的余数，结果在[0, m-1]范围内。
```

### 2.2 模运算性质
```
(a + b) mod m = [(a mod m) + (b mod m)] mod m
(a - b) mod m = [(a mod m) - (b mod m)] mod m
(a × b) mod m = [(a mod m) × (b mod m)] mod m
a^b mod m = [(a mod m)^b] mod m
```

### 2.3 快速幂（模幂运算）
```python
def pow_mod(a, b, m):
    """计算 (a^b) mod m，使用快速幂算法"""
    result = 1
    a = a % m
    while b > 0:
        if b % 2 == 1:
            result = (result * a) % m
        a = (a * a) % m
        b = b // 2
    return result
```

---

## 3. 同余

### 3.1 同余定义
```
若a ≡ b mod m，则称a与b模m同余。
等价于：m | (a - b)
```

### 3.2 同余性质
```
若a ≡ b mod m，c ≡ d mod m，则：
a + c ≡ b + d mod m
a - c ≡ b - d mod m
a × c ≡ b × d mod m
```

### 3.3 模逆元
```
若a × x ≡ 1 mod m，则称x是a在模m下的逆元，记作a^{-1}。
a有逆元当且仅当gcd(a, m) = 1。
```

---

## 4. 欧几里得算法

### 4.1 欧几里得算法（求最大公约数）
```
gcd(a, b) = gcd(b, a mod b)
gcd(a, 0) = a
```

### 4.2 Python实现
```python
def gcd(a, b):
    """欧几里得算法求最大公约数"""
    while b != 0:
        a, b = b, a % b
    return a

def lcm(a, b):
    """求最小公倍数"""
    return a * b // gcd(a, b) if a and b else 0
```

### 4.3 扩展欧几里得算法
```
求整数x, y，使得ax + by = gcd(a, b)
```

```python
def extended_gcd(a, b):
    """扩展欧几里得算法"""
    if b == 0:
        return a, 1, 0
    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y

def mod_inverse(a, m):
    """求a在模m下的逆元"""
    g, x, y = extended_gcd(a, m)
    if g != 1:
        return None  # 逆元不存在
    return x % m
```

---

## 5. 中国剩余定理

### 5.1 中国剩余定理（CRT）
```
给定两两互质的正整数m₁, m₂, ..., m_k，求x满足：
x ≡ a₁ mod m₁
x ≡ a₂ mod m₂
...
x ≡ a_k mod m_k

解：x ≡ Σ a_i × M_i × y_i mod M
其中：
M = m₁ × m₂ × ... × m_k
M_i = M / m_i
y_i ≡ M_i^{-1} mod m_i
```

### 5.2 Python实现
```python
def chinese_remainder_theorem(a_list, m_list):
    """中国剩余定理求解"""
    # 验证m_list是否两两互质
    for i in range(len(m_list)):
        for j in range(i + 1, len(m_list)):
            if gcd(m_list[i], m_list[j]) != 1:
                return None
    
    M = 1
    for m in m_list:
        M *= m
    
    result = 0
    for a, m in zip(a_list, m_list):
        Mi = M // m
        yi = mod_inverse(Mi, m)
        result += a * Mi * yi
        result %= M
    
    return result
```

---

## 6. 欧拉函数与欧拉定理

### 6.1 欧拉函数φ(n)
```
φ(n)：小于等于n且与n互质的正整数的个数。

若n = p₁^{k₁} × p₂^{k₂} × ... × p_m^{k_m}
则φ(n) = n × Π (1 - 1/p_i)
```

### 6.2 欧拉函数性质
```
若m, n互质，则φ(mn) = φ(m) × φ(n)
若p是素数，则φ(p) = p - 1
若p是素数，则φ(p^k) = p^k - p^{k-1}
```

### 6.3 Python实现
```python
def euler_phi(n):
    """计算欧拉函数φ(n)"""
    result = n
    i = 2
    while i * i <= n:
        if n % i == 0:
            while n % i == 0:
                n = n // i
            result -= result // i
        i += 1
    if n > 1:
        result -= result // n
    return result
```

### 6.4 欧拉定理
```
若a与n互质，则a^φ(n) ≡ 1 mod n
```

### 6.5 费马小定理
```
若p是素数，且a不被p整除，则a^(p-1) ≡ 1 mod p
推论：若p是素数，则a^p ≡ a mod p（对任意a）
```

---

## 7. 素性测试

### 7.1 朴素素性测试
```python
def is_prime_simple(n):
    """朴素素性测试"""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True
```

### 7.2 Miller-Rabin素性测试（概率性测试）
```python
def is_prime_miller_rabin(n, k=5):
    """Miller-Rabin素性测试"""
    if n <= 1:
        return False
    elif n <= 3:
        return True
    elif n % 2 == 0:
        return False
    
    # 将n-1表示为d*2^s
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    
    # 进行k次测试
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for a in bases[:k]:
        if a >= n:
            continue
        x = pow_mod(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = pow_mod(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True
```

---

## 8. 数论在密码学中的应用

### 8.1 RSA加密算法
```
RSA基于大数分解困难问题。

密钥生成：
1. 选择两个大素数p, q
2. 计算n = p × q
3. 计算φ(n) = (p - 1) × (q - 1)
4. 选择公钥e，使得1 < e < φ(n)且gcd(e, φ(n)) = 1
5. 计算私钥d ≡ e^{-1} mod φ(n)

加密：c = m^e mod n
解密：m = c^d mod n
```

### 8.2 Python简化实现
```python
def rsa_key_generation(p, q):
    """RSA密钥生成（简化版）"""
    n = p * q
    phi_n = (p - 1) * (q - 1)
    
    # 选择e
    e = 65537  # 常用的公钥指数
    while gcd(e, phi_n) != 1:
        e += 2
    
    d = mod_inverse(e, phi_n)
    return (e, n), (d, n)

def rsa_encrypt(m, public_key):
    """RSA加密"""
    e, n = public_key
    return pow_mod(m, e, n)

def rsa_decrypt(c, private_key):
    """RSA解密"""
    d, n = private_key
    return pow_mod(c, d, n)
```

---

## 本章小结

数论是密码学和算法设计的重要基础，本章要点：
1. 素数、整除、质因数分解
2. 模运算与同余
3. 欧几里得算法与扩展欧几里得算法
4. 中国剩余定理
5. 欧拉函数、欧拉定理、费马小定理
6. 素性测试
7. RSA加密等密码学应用

