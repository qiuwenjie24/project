
#-------------下载数据集到指定目录------------
import hashlib
import os
import requests

def download(name, cache_dir='./house_data'):  #@save
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

download('kaggle_house_train')
download('kaggle_house_test')




#-------------加载数据------------
import pandas as pd
train_data = pd.read_csv('./house_data/kaggle_house_pred_train.csv')
test_data = pd.read_csv('./house_data/kaggle_house_pred_test.csv')


# print(train_data.shape)
# print(test_data.shape)
# print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
# print(all_features.iloc[0:4, [0, 1, 2, -2, -1]])
# print(all_features.shape)


#----------对数据预处理----------
import torch
# 提取数值型特征
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# print(numeric_features.shape)
# 标准化数据
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std())) 
# 填充缺失值  
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 用独热编码表示类别型特征
# print(all_features.shape)
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features = all_features.astype('float64')
# print(all_features.dtypes)
# print(all_features.shape)

# 将数据集分成训练集和测试集
n_train = train_data.shape[0]

train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)


#----------训练----------------
from torch import nn
from torch.utils import data 

# 输入特征的数量
in_features = train_features.shape[1]
# 定义模型(线性模型)
def get_net(): 
    net = nn.Sequential(nn.Linear(in_features,1))
    return net 


# 使用MSE损失函数，用于训练优化
loss = nn.MSELoss()
# 考虑相对误差而不是绝对误差，因此使用logRMSE，只用于表示当前损失的程度，不用于优化
def log_rmse(net, features, labels):
    # 为了保持对数计算的稳定，将小于1e-6的数设置为1e-6
    preds = torch.max(net(features), torch.tensor(1e-6))
    labels = torch.max(labels, torch.tensor(1e-6))
    rmse = torch.sqrt(loss(torch.log(preds),
                           torch.log(labels)))
    return rmse.item() 


# 训练数据加载器(迭代器)
def load_data_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)



# 使用Adam优化算法训练 
def train(net, train_features, train_labels, test_features, test_labels,
    num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = load_data_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, 
                                 weight_decay=weight_decay)

    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls  


# ---------------K折交叉验证------------------
import matplotlib.pyplot as plt
# 获取当前折的训练数据和测试数据
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k 
    X_train, y_train =None, None 
    for j in range(k):
        idx = slice(j * fold_size, (j+1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

# K折交叉验证中训练
from d2l import torch as d2l
import numpy as np
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, 
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    train_l_aver, valid_l_aver = [0]*(num_epochs + 1), [0]*(num_epochs + 1)
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, 
                                   weight_decay, batch_size)

        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        train_l_aver = [x + y/k for x, y in zip(train_l_aver, train_ls)]
        valid_l_aver = [x + y/k for x, y in zip(valid_l_aver, valid_ls)]
        if i == 0:
            # 画图，i=0训练损失和验证损失
            plt.figure() 
            plt.plot(list(range(1, num_epochs + 1)), train_ls, label='train', linestyle='-', color='blue')
            plt.plot(list(range(1, num_epochs + 1)), valid_ls, label='valid', linestyle='-', color='red')
            plt.yscale("log")
            plt.legend()
            plt.xlabel("epoch")
            plt.ylabel("log RMSE")
            plt.title('0-th data as valid_data')
            plt.grid(True)  # 显示网格
            # plt.show()
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')

    plt.figure() 
    plt.plot(list(range(1, num_epochs + 1)), train_l_aver, label='train', linestyle='-', color='blue')
    plt.plot(list(range(1, num_epochs + 1)), valid_l_aver, label='valid', linestyle='-', color='red')
    plt.yscale("log")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("log RMSE")
    plt.title('average loss')
    plt.grid(True)  # 显示网格  
    return train_l_sum / k, valid_l_sum / k 



# ---------超参数选择，找到损失最小的一组超参数-----------
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证：平均训练log rmse: {float(train_l):f}'
      f'平均验证log rmse: {float(valid_l):f}')





# --------------预测---------------
# 用全部数据训练并预测
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    # 画图，训练损失
    plt.figure() 
    plt.plot(np.arange(1, num_epochs + 1), train_ls, label='tran')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('log RMSE')
    plt.grid(True)  # 显示网格 
    # plt.show()

    print(f'训练log rmse: {float(train_ls[-1]):f}')
    # 应用于测试集
    preds = net(test_features).detach().numpy()
    # 导出预测数据csv
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)

train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)

plt.show()

