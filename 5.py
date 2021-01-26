#!/usr/bin/env python
# coding: utf-8

# In[5]:


#1 
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np

for i in range(0, 7):
    a = input()
    red = [1, 3 ,5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36]
    black = [2, 4, 6, 8, 10, 11, 13 ,15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35]
    x = np.random.randint(0, 37)
    if x==0:
        print("0")
    elif x in black:
        print("черный")
    elif x in red:
        print("красный")
        


# In[7]:


#2.1
k,m = 0, 0
n = 100
for i in range(0, n):
    x = np.random.uniform(0, 10)
    if x<5:
#        print("орел")
        k = k + 1
    else:
#        print("решка")
        m = m + 1
p1 = k / n
p2 = m / n
print (p1 + p2)


# In[17]:


#3.1
import math
k, n = 0, 10000
a = np.random.randint(0, 2, n)
b = np.random.randint(0, 2, n)
c = np.random.randint(0, 2, n)
d = np.random.randint(0, 2, n)
x = a + b + c + d
for i in range(0, n):
    if x[i] == 2:
        k = k + 1
#print(a, b, c, d)
#print(x)
print(k, n, k/n)

c42 = math.factorial(4)/(math.factorial(2) * math.factorial(4-2))
p = c42 * ((0.5 ** 2) * ((1-0.5) ** (4-2)))
print(p)


# In[18]:


#4
import itertools
for p in itertools.permutations("012345",4):
    print(''.join(str(x) for x in p))


# In[19]:


for p in itertools.combinations("012345",4):
    print(''.join(p))


# In[28]:


#5 что такое r?
import matplotlib.pyplot as plt
n = 100
r = 0.7
x = np.random.rand(n)
y = r*x + (1 - r)*np.random.rand(n)
plt.plot(x, y, 'o')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

a = (np.sum(x)*np.sum(y) - n*np.sum(x*y))/(np.sum(x)*np.sum(x) - n*np.sum(x*x))
b = (np.sum(y) - a*np.sum(x))/n
rc = np.sum((x - np.mean(x)) * (y - np.mean(y))) / math.sqrt(np.sum((x - np.mean(x)) ** 2) * np.sum((y - np.mean(y)) ** 2))

print(a, b, rc)

plt.plot([0, 1], [b, a + b])
plt.show()
c = np.corrcoef(x, y)
print(c)


# In[ ]:




