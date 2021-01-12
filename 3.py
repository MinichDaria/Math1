#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Часть 1
from math import sqrt
a, b, c = map(float, input('Введите координаты вектора через пробел: ').split())
vlenth = sqrt(a ** 2 + b ** 2 + c **2)
print (round(vlenth, 5))


# In[5]:


# как сделать больше точек и почему единичные отрезки такие разные?
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import math
import numpy as np
x = np.linspace(-6, 6, 201)
plt.grid(True)
r = 5
x = []
y = []
y1 = []
for i in range(-5, 6):
    x_ = i
    x.append(x_)
    y.append(math.sqrt(r ** 2 - x_ ** 2))
    y1.append(-math.sqrt(r ** 2 - x_ ** 2))
plt.plot(x,y)
plt.plot(x,y1)


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
fig = figure()
ax = Axes3D(fig)
X = np.arange(-5, 5, 2)
Y = np.arange(-50, 50, 20)
X, Y = np.meshgrid(X, Y)
Z = 2*X + 3*Y
Z2 = 2*X + 3*Y + 100
ax.plot_surface(X, Y, Z)
ax.plot_surface(X, Y, Z2)


# In[8]:


# Часть 2
x = np.linspace(-2*np.pi, 3*np.pi, 201)
plt.plot(x, 3 * np.cos(x - 5) + 1)
plt.plot(x, 10 * np.cos(x + 2) + 8)


# In[10]:


r = int(input())
a = int(input())
x = r * np.cos(a)
y = r * np.sin(a)
print (x, y)


# In[15]:


from scipy.optimize import fsolve
def equations(p):
    x, y = p
    return (x**2 - 1 - y, np.exp(x) + x * (1 - y) - 1)
x1, y1 = fsolve(equations, (-1, 1))
x2, y2 = fsolve(equations, (2, 4))
print (x1, y1)
print (x2, y2)
x = np.linspace(-2, 3, 201)
plt.plot(x, x ** 2 - 1)
plt.plot(x, 1 - ((1 - np.exp(x)) / x))


# In[ ]:




