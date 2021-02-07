#!/usr/bin/env python
# coding: utf-8

# In[2]:


#1
import numpy as np
import matplotlib.pyplot as plt
A = np.array([[1, 2, 3], [4, 0, 6], [7, 8, 9]])
B = np.array([12, 2, 1])
np.linalg.solve(A, B)


# In[3]:


#2
A = np.array([[1, 2, -1], [3, -4, 0], [8, -5, 2], [2, 0, -5], [11, 4, -7]])
B = np.array([1, 7, 12, 7, 15])
np.linalg.lstsq(A, B)


# In[4]:


#3
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([2, 5, 1])
np.linalg.solve(A, B)


# In[5]:


A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([[2, 5, 1]])
C = np.concatenate((A,B.T), axis=1)
print (C)
np.linalg.matrix_rank(A, 0.0001), np.linalg.matrix_rank(C, 0.0001)


# In[14]:


A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([[12, 15, 18]])
C = np.concatenate((A,B.T), axis=1)
print (C)
np.linalg.matrix_rank(A, 0.0001), np.linalg.matrix_rank(C, 0.0001)
np.linalg.solve(A, B)


# In[15]:


#4
import scipy 
import scipy.linalg 
A = np.array([ [1, 2, 3], [2, 16, 21], [4, 28, 73] ])
B = np.array([1, 2, 3])
P, L, U = scipy.linalg.lu(A)

print(P)
print(L)
print(U)
np.linalg.solve(A, B)


# In[22]:


#5
import matplotlib.pyplot as plt
def Q(x, y, z):
    return (x**2 + y**2 + z**2)

x = np.linspace(-1, 5, 2001)
plt.plot(x, Q(x, 1 - 2 * y + 2, (5 * y - 2 * z + 12) / 8)

plt.grid(True)
plt.show()


# In[ ]:


#6
A = np.array([ [1, 2, 3], [4, 5, 6], [7, 8, 9] ])
B = np.array([2, 5, 11])
Q, R = np.linalg.qr(A)


# In[ ]:




