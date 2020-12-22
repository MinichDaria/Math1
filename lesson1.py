#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt


# In[8]:


x = np.linspace(-np.pi, np.pi, 201)
plt.plot(x, np.cos(x))
plt.plot(x, 3*np.cos(x))
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[ ]:




