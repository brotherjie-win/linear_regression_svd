#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 数据预处理/打标签
data_url = r"effect_validation_dataset\boston.csv"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
house_df = pd.DataFrame(data)
house_targe = pd.DataFrame(target)
house_df.columns = [['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptradio','b','lstat']]
house_targe.columns = [['medv'] ]

# 数据集划分(训练集:测试集=7:3)
X_train, X_test, Y_train, Y_test = train_test_split(house_df, house_targe,random_state=23)


# In[3]:


X_train


# In[12]:


Y_train


# In[14]:


X_test


# In[16]:


Y_test


# In[22]:


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


# In[23]:


def cal_mse(y_predict, y_true):
    assert y_predict.ndim == y_true.ndim, 'y_predict和y_true需要维度相同'
    return np.mean((y_predict - y_true) ** 2)


# In[25]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print('=====方法1：本文方法=====')
# SELF
lr = LR(fit_intercept=True)
lr.fit(X_train.values,Y_train.values)
Y_train_predict = lr.predict(X_train.values)
Y_test_predict = lr.predict(X_test.values)
print('MSE in training set:%.4f'%(mean_squared_error(Y_train.values, Y_train_predict)))
print('MSE in testing set:%.4f' % (mean_squared_error(Y_test.values, Y_test_predict)))
print('MAE in training set:%.4f'%(mean_absolute_error(Y_train.values, Y_train_predict)))
print('MAE in testing set:%.4f' % (mean_absolute_error(Y_test.values, Y_test_predict)))
print('RMSE in training set:%.4f'%(np.sqrt(mean_absolute_error(Y_train.values, Y_train_predict))))
print('RMSE in testing set:%.4f' % (np.sqrt(mean_absolute_error(Y_test.values, Y_test_predict))))

# SK
lr_sk = LinearRegression()
lr_sk.fit(X_train.values, Y_train.values)
Y_train_predict_sk = lr_sk.predict(X_train.values)
Y_test_predict_sk = lr_sk.predict(X_test.values)
print('=====Sklearn Linear Regression=====')
print('MSE in training set:%.4f'%(mean_squared_error(Y_train.values, Y_train_predict_sk)))
print('MSE in testing set:%.4f' % (mean_squared_error(Y_test.values, Y_test_predict_sk)))
print('MAE in training set:%.4f'%(mean_absolute_error(Y_train.values, Y_train_predict_sk)))
print('MAE in testing set:%.4f' % (mean_absolute_error(Y_test.values, Y_test_predict_sk)))
print('RMSE in training set:%.4f'%(np.sqrt(mean_absolute_error(Y_train.values, Y_train_predict_sk))))
print('RMSE in testing set:%.4f' % (np.sqrt(mean_absolute_error(Y_test.values, Y_test_predict_sk))))

