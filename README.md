# Machine-learning
#Machine learning code written in Python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv(r"E:\不重要\regress_data1.csv",header=None)
#数据归一化函数(min-max归一化)
def min_max_normalization(x_):#传入需要归一化的数据,x为列向量,由于x第一列是截距项1，因此跳过这一列
    xx=x_.copy()
    # 求出x有多少组数据
    x_row = xx.shape[0]
    x_col = xx.shape[1]
    flag=0
    for i in range(x_row):
        if xx[i,0]==1:
            flag=0
        else:#如果第一列不是截距项
            flag=1

    if flag==0:#如果第一列是截距项
        # 求出x有多少个特征
        x_col = x_col - 1
        for col in range(1, x_col + 1):
            xmin = np.min(xx[:, col])
            xmax = np.max(xx[:, col])
            for row in range(x_row):
                xx[row, col] = (xx[row, col] - xmin) / (xmax - xmin)
    else:#如果不是截距项
        for col in range(x_col):
            xmin=np.min(xx[:,col])
            xmax=np.max(xx[:,col])
            for row in range(x_row):
                xx[row,col]=(xx[row,col]-xmin)/(xmax-xmin)
    return xx
alpha=0.0001
x=data.iloc[1:,0].values
y=data.iloc[1:,1].values
#转换为numpy数组
x=np.array(x,dtype=float)
y=np.array(y,dtype=float)

plt.scatter(x,y)
x_min=x.min()
x_max=x.max()
x=x.reshape(-1,1)#将x转换为列向量
y=y.reshape(-1,1)#y
#添加截距项
one=np.ones((x.shape[0],1))
x=np.hstack((one,x))
n=x.shape[1]
#设置beta参数项
beta_=np.zeros((n,n-1))#-1是因为增加了截距项
beta1=beta_.copy()
beta2=beta_.copy()
#梯度下降

for i in range(1000):
    delta=np.dot(1/x.shape[0]*x.T,np.dot(x,beta1)-y)
    beta1=beta1-alpha*delta

beta0_=beta1[0,0]
beta1_=beta1[1,0]
x1=np.linspace(x_min,x_max,100)
y1=beta0_+beta1_*x1
plt.plot(x1,y1,color='red',label='gradient descent')


#对梯度下降法引入L2范数(L2正则化)

lambda_=1.0#正则化强度
L_beta=np.zeros(1000)#初始化损失函数数组
for i in range(1000):
    #计算梯度
    re=np.dot(1/x.shape[0]*x.T,np.dot(x,beta2)-y)
    #引入L2范数
    delta1=re+2*lambda_*beta2
    #更新beta2
    beta2=beta2-alpha*delta1
    # 计算y-x*beta
    y_xbeta = y - np.dot(x, beta2)
    L_beta[i] = (1 / 1000) * (np.dot(y_xbeta.T, y_xbeta)).item()+lambda_*np.dot(beta2.T,beta2).item()

beta11=beta2[0,0]
beta22=beta2[1,0]
x11=np.linspace(x_min,x_max,100)
y11=beta11+beta22*x11
plt.plot(x11,y11,color='yellow',label='add L2')
#最小二乘法(前提XTX可逆)
beta_least_square=np.dot(np.linalg.inv(np.dot(x.T,x)),np.dot(x.T,y))
#print('beta_least_square:',beta_least_square)
beta_least_square1=beta_least_square[0,0]
beta_least_square2=beta_least_square[1,0]
x3=np.linspace(x_min,x_max,100)
y3=beta_least_square1+x3*beta_least_square2
plt.plot(x3,y3,color='blue',label='least square')
plt.legend()
plt.show()


#最小二乘法引入数据归一化(min-max)
x_normalization=min_max_normalization(x)
x_normalized=x_normalization[:,1]
x_normalized=x_normalized.reshape(-1,1)
#对y也归一化处理
y_normalization=min_max_normalization(y)
print(x_normalization)
#显示(仅限二维)
plt.scatter(x_normalized,y_normalization)
x_normalization_min=np.min(x_normalization[:,1])
x_normalization_max=np.max(x_normalization[:,1])
#最小二乘法
beta_normalization=np.dot(np.linalg.inv(np.dot(x_normalization.T,x_normalization)),np.dot(x_normalization.T,y_normalization))
beta_normalization1=beta_normalization[0,0]
beta_normalization2=beta_normalization[1,0]
x4=np.linspace(x_normalization_min,x_normalization_max,100)
y4=beta_normalization1+beta_normalization2*x4
plt.plot(x4,y4,color='red',label='normalization')
plt.legend()
plt.show()

#导入regress_data2
data2=pd.read_csv(r"E:\不重要\regress_data2.csv",encoding='gbk')
print(data2)
x1_data=data2.iloc[:,0].values
#转换为numpy数组
x1_data=np.array(x1_data,dtype=float)
x2_data=data2.iloc[:,1].values
x2_data=np.array(x2_data,dtype=float)
y_data=data2.iloc[:,2].values
y_data=np.array(y_data,dtype=float)
#拷贝x2_data
x2_data_copy=x2_data.copy()
#将x1_data归一化为x1_data_normalized
x1_data_normalized=x1_data.copy()
x1_data_min=np.min(x1_data)
x1_data_max=np.max(x1_data)
for i in range(x1_data.shape[0]):
    x1_data_normalized[i]=(x1_data_normalized[i]-x1_data_min)/(x1_data_max-x1_data_min)
#将y_data归一化为y_data_normalized
y_data_normalized=y_data.copy()
y_data_min=np.min(y_data)
y_data_max=np.max(y_data)
for ii in range(y_data.shape[0]):
    y_data_normalized[ii]=(y_data_normalized[ii]-y_data_min)/(y_data_max-y_data_min)
#将x1_data_normalized和x2_data_copy,y_data_normalized转换为列向量
x1_data_normalized=x1_data_normalized.reshape(-1,1)
x2_data_copy=x2_data_copy.reshape(-1,1)
y_data_normalized=y_data_normalized.reshape(-1,1)
#合并x1_data_normalized,x2_data_copy
x_=np.hstack((x1_data_normalized,x2_data_copy))
#在左边加入截距项
one_data=np.ones((x1_data.shape[0],1))
x_=np.hstack((one_data,x_))
#初始化beta_data
beta_data=np.zeros((x_.shape[1],1))
#定义正则化强度lamda_data=1.1,学习率alpha_data=0.01
lamda_data=1.1
alpha_data=0.001
#初始化损失函数数组
L_data2_beta=np.zeros(1000)
#迭代次数1000次
for it in range(1000):
    beta_data=beta_data-alpha_data*(np.dot((1/x_.shape[0])*(x_.T),(np.dot(x_,beta_data)-y_data_normalized))+2*lamda_data*beta_data)
    #计算y-x*beta
    y_data2_xbeta=y_data_normalized-np.dot(x_,beta_data)
    L_data2_beta[it]=(1/1000)*(np.dot(y_data2_xbeta.T,y_data2_xbeta)).item()+lamda_data*np.dot(beta_data.T,beta_data).item()
print(beta_data)
beta_data0=beta_data[0,0]
beta_data1=beta_data[1,0]
beta_data2=beta_data[2,0]
print(f"y={beta_data0}+{beta_data1}x1+{beta_data2}x2")
#画出损失曲线1
x_L=np.linspace(0,1000,1000)
plt.plot(x_L,L_beta,label='Loss curve of data')
plt.legend()
plt.show()
#画出损失曲线2
x_L_data2=np.linspace(0,1000,1000)
plt.plot(x_L_data2,L_data2_beta,label='Loss curve of data2')
plt.legend()
plt.show()
