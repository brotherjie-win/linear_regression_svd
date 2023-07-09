#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
X_inv,Y_inv=make_regression(n_samples=100, n_features=10, n_informative=10, n_targets=1, noise=1.2, random_state=2023)
X_ninv,Y_ninv=make_regression(n_samples=10, n_features=100, n_informative=100, n_targets=1, noise=1.2, random_state=2023)
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


# In[5]:


X_inv


# In[7]:


Y_inv


# In[9]:


X_ninv


# In[11]:


Y_ninv


# In[13]:


X_inv_train, X_inv_test, Y_inv_train, Y_inv_test = train_test_split(X_inv, Y_inv, test_size=0.3, random_state=2023)
X_ninv_train, X_ninv_test, Y_ninv_train, Y_ninv_test = train_test_split(X_ninv, Y_ninv, test_size=0.3, random_state=2023)


# In[15]:


X_inv_train


# In[17]:


X_inv_test


# In[19]:


Y_inv_train


# In[21]:


Y_inv_test


# In[23]:


def cal_mse(y_predict, y_true):
    assert y_predict.ndim == y_true.ndim, 'y_predict和y_true需要维度相同'
    return np.mean((y_predict - y_true) ** 2)


# In[24]:


class LR(object):
    def __init__(self,fit_intercept=True):
        """
        :param fit_intercept: 是否加入截距项，如果加入，需要对X第一列增广，即拼接上全1向量
        """
        self.beta = None #线性回归参数
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        :param X: 样本矩阵
        """
        #当有截距项，需要在X左侧增广，加上一列全1的列向量
        if self.fit_intercept:
            X = np.hstack((np.ones_like(y.reshape((-1,1))), X))

        ##判断(XTX)是否可逆
        n_sample = X.shape[0]
        n_feature = X.shape[1]

        #简化处理：特征数量大于样本数量时矩阵显然不可逆
        if n_feature > n_sample:
            is_inv = False
        #进一步判断行列式
        elif np.linalg.det(np.matmul(X.T, X)) == 0:
            is_inv = False
        else:
            is_inv = True

        #可逆
        if is_inv:
            self.beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        #不可逆
        else:
            u,s,vt = np.linalg.svd(X) #SVD分解,s是向量，表示奇异值
            if len(np.nonzero(s)[0]) == X.shape[0]:
                sigma_inv_vector = 1 / s
            else:#当出现0奇异值，1/s会报错，另外处理
                n_nonzero = len(np.nonzero(s)[0])
                s_nonzero = s[:n_nonzero]
                s_inv = 1 / s #对角阵的伪逆
                zeros = np.zeros((n_feature - len(s_inv)))
                sigma_inv_vector = np.hstack((s_inv,zeros))
            sigma_inv_diag = np.diag(sigma_inv_vector)
            if X.shape[0] == X.shape[1]: #sigma是方阵(行=列)
                sigma_inv = sigma_inv_diag
            elif X.shape[0] > X.shape[1]: #sigma是竖的矩形(行>列)
                sigma_zeros = np.zeros((X.shape[1],(X.shape[0] - X.shape[1])))
                sigma_inv = np.hstack((sigma_inv_diag, sigma_zeros))
            else:#sigma是横的矩形(行<列)
                sigma_zeros = np.zeros(((X.shape[1] - X.shape[0]),X.shape[0]))
                sigma_inv = np.vstack((sigma_inv_diag, sigma_zeros))

            self.beta = vt.T @ sigma_inv @ u.T @ y

        self.beta = self.beta.reshape((-1,1))

    def predict(self, X):
        """
        :param X: 测试集
        :return: y_predict
        """
        if X.shape[1] != self.beta.shape[0]:
            X = np.hstack((np.ones((X.shape[0], 1)), X))

        y_predict = X.dot(self.beta)
        return y_predict


# In[25]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print('=====实验2：不可逆情形=====')
# SELF
lr = LR(fit_intercept=True)
lr.fit(X_ninv_train,Y_ninv_train)
Y_ninv_train_predict = lr.predict(X_ninv_train)
Y_ninv_test_predict = lr.predict(X_ninv_test)
print('=====本文方法=====')
print('MSE in training set:%.4f'%(mean_squared_error(Y_ninv_train, Y_ninv_train_predict)))
print('MSE in testing set:%.4f' % (mean_squared_error(Y_ninv_test, Y_ninv_test_predict)))
print('MAE in training set:%.4f'%(mean_absolute_error(Y_ninv_train, Y_ninv_train_predict)))
print('MAE in testing set:%.4f' % (mean_absolute_error(Y_ninv_test, Y_ninv_test_predict)))
print('RMSE in training set:%.4f'%(np.sqrt(mean_absolute_error(Y_ninv_train, Y_ninv_train_predict))))
print('RMSE in testing set:%.4f' % (np.sqrt(mean_absolute_error(Y_ninv_test, Y_ninv_test_predict))))

# SK
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
lr_sk = LinearRegression()
lr_sk.fit(X_ninv_train,Y_ninv_train)
Y_ninv_train_predict_sk = lr_sk.predict(X_ninv_train)
Y_ninv_test_predict_sk = lr_sk.predict(X_ninv_test)
print('=====Sklearn方法=====')
print('MSE in training set:%.4f'%(mean_squared_error(Y_ninv_train, Y_ninv_train_predict_sk)))
print('MSE in testing set:%.4f' % (mean_squared_error(Y_ninv_test, Y_ninv_test_predict_sk)))
print('MAE in training set:%.4f'%(mean_absolute_error(Y_ninv_train, Y_ninv_train_predict_sk)))
print('MAE in testing set:%.4f' % (mean_absolute_error(Y_ninv_test, Y_ninv_test_predict_sk)))
print('RMSE in training set:%.4f'%(np.sqrt(mean_absolute_error(Y_ninv_train, Y_ninv_train_predict_sk))))
print('RMSE in testing set:%.4f' % (np.sqrt(mean_absolute_error(Y_ninv_test, Y_ninv_test_predict_sk))))
print("========")
print('SVD LR coef 2-norm:%.4f'%np.linalg.norm(lr.beta))
print('sklearn LR coef 2-norm:%.4f' % np.linalg.norm(np.hstack((lr_sk.coef_, lr_sk.intercept_))))


# In[26]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print('=====实验1：可逆情形=====')
# SELF
lr = LR(fit_intercept=True)
lr.fit(X_inv_train,Y_inv_train)
Y_inv_train_predict = lr.predict(X_inv_train)
Y_inv_test_predict = lr.predict(X_inv_test)
print('=====本文方法=====')
print('MSE in training set:%.4f'%(mean_squared_error(Y_inv_train, Y_inv_train_predict)))
print('MSE in testing set:%.4f' % (mean_squared_error(Y_inv_test, Y_inv_test_predict)))
print('MAE in training set:%.4f'%(mean_absolute_error(Y_inv_train, Y_inv_train_predict)))
print('MAE in testing set:%.4f' % (mean_absolute_error(Y_inv_test, Y_inv_test_predict)))
print('RMSE in training set:%.4f'%(np.sqrt(mean_absolute_error(Y_inv_train, Y_inv_train_predict))))
print('RMSE in testing set:%.4f' % (np.sqrt(mean_absolute_error(Y_inv_test, Y_inv_test_predict))))

# SK
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
lr_sk = LinearRegression()
lr_sk.fit(X_inv_train,Y_inv_train)
Y_inv_train_predict_sk = lr_sk.predict(X_inv_train)
Y_inv_test_predict_sk = lr_sk.predict(X_inv_test)
print('=====Sklearn方法=====')
print('MSE in training set:%.4f'%(mean_squared_error(Y_inv_train, Y_inv_train_predict_sk)))
print('MSE in testing set:%.4f' % (mean_squared_error(Y_inv_test, Y_inv_test_predict_sk)))
print('MAE in training set:%.4f'%(mean_absolute_error(Y_inv_train, Y_inv_train_predict_sk)))
print('MAE in testing set:%.4f' % (mean_absolute_error(Y_inv_test, Y_inv_test_predict_sk)))
print('RMSE in training set:%.4f'%(np.sqrt(mean_absolute_error(Y_inv_train, Y_inv_train_predict_sk))))
print('RMSE in testing set:%.4f' % (np.sqrt(mean_absolute_error(Y_inv_test, Y_inv_test_predict_sk))))

