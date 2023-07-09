#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


dataset_loc = r"effect_validation_dataset/xtrain.npy"


# In[3]:


X_train = np.load(dataset_loc)


# In[6]:


np.set_printoptions(
    infstr='inf',
    nanstr='nan',
    formatter=None,
    precision=2,    # 精度，保留小数点后几位
    threshold=500,
    # 最多可现实的Array元素个数
    # 限制的是基本元素个数，如3*5的矩阵，限制的是15而非3（行）
    # 如果超过就采用缩略显示
    edgeitems=3,
    # 在缩率显示时在起始和默认显示的元素个数
    linewidth=150,  # 每行最多显示的字符数，默认80，超过则换行显示
    suppress=True   # 浮点显示（不用科学计数法）
)


# In[7]:


print(X_train)


# In[10]:


print("X_train的数据总数：%d" % X_train.size)
print("X_train的行数(样本点数)：%d" % X_train.shape[0])
print("X_train的列数(变量数/维度)：%d" % X_train.shape[1])


# In[13]:


print("X_train的第135行第9列为：%f" % X_train[134, 8])
print("X_train的第3行第12列为：%f" % X_train[2, 11])

