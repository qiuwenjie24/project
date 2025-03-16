
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
    r = requests.get(url, stream=True)
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

#----------对数据预处理----------
from scipy.stats import cauchy
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np

def detect_outliers(data):
    """使用柯西分布检测异常值"""
    params = cauchy.fit(data['SalePrice'])
    threshold = params[0] + 3*params[1]
    return data[data['SalePrice'] > threshold].index

# 异常值检测
outlier_idx = detect_outliers(train_data)
train_data = train_data.drop(outlier_idx)  # 删除异常样本
print(f"The number of abnormal samples deleted: {len(outlier_idx)}")

# 将数据集分成训练集、测试集和标签，丢弃id列
train_features = train_data.iloc[:, 1:-1]
train_labels = train_data.iloc[:, -1]
test_features = test_data.iloc[:, 1:]

# 提取数值型特征和类别型特征的索引
num_features = train_features.dtypes[train_features.dtypes != 'object'].index
cat_features = train_features.dtypes[train_features.dtypes == 'object'].index

# 计算每个特征的缺失值比例
num_missing_ratio = train_features[num_features].isnull().mean()
num_sorted_list = sorted(enumerate(num_missing_ratio), key=lambda x: x[1], reverse=True)
print(f'First 4 maximal missing ratios in numerical features: {[f"{y * 100:.2f}%" for _, y in num_sorted_list[:4]]}')

cat_missing_ratio = train_features[cat_features].isnull().mean()
cat_sorted_list = sorted(enumerate(cat_missing_ratio), key=lambda x: x[1], reverse=True)
print(f'First 5 maximal missing ratios in category features: {[f"{y * 100:.2f}%" for _, y in cat_sorted_list[:7]]}')

cat_indix_H = [x[0] for x in cat_sorted_list[:3]]   # 高缺失值比例的特征索引
cat_indix_M = [x[0] for x in cat_sorted_list[3:6]]  # 中缺失值比例的特征索引
cat_indix_L = [x[0] for x in cat_sorted_list[6:]]   # 低缺失值比例的特征索引

cat_features_H = train_features[cat_features].columns[cat_indix_H]   # 相应的特征名称
cat_features_M = train_features[cat_features].columns[cat_indix_M]   
cat_features_L = train_features[cat_features].columns[cat_indix_L]
train_features.drop(columns=cat_features_H, inplace=True) # 删除高缺失值列，并修改原数据


# 缺失值处理，数值型填充均值，类别型填充众数或保留或删除
num_imputer = SimpleImputer(strategy='mean')  
cat_imputer_L = SimpleImputer(strategy='most_frequent')
train_features[num_features] = num_imputer.fit_transform(train_features[num_features])
train_features[cat_features_L] = cat_imputer_L.fit_transform(train_features[cat_features_L])

test_features[num_features] = num_imputer.transform(test_features[num_features])
test_features[cat_features_L] = cat_imputer_L.transform(test_features[cat_features_L])

# 特征工程
train_features['YearSin'] = np.sin(2 * np.pi * train_features['YearBuilt'] / 100)
train_features['YearCos'] = np.cos(2 * np.pi * train_features['YearBuilt'] / 100)
test_features['YearSin'] = np.sin(2 * np.pi * test_features['YearBuilt'] / 100)
test_features['YearCos'] = np.cos(2 * np.pi * test_features['YearBuilt'] / 100)

# 标准化数值型特征
num_features = num_features.drop(['YearSin', 'YearCos'], errors='ignore')  # 从标准化列中移除'YearSin'和'YearCos'
scaler = StandardScaler()
train_features[num_features] = scaler.fit_transform(train_features[num_features])
test_features[num_features] = scaler.transform(test_features[num_features])

# 显式指定需要处理的列，用独热编码表示类别型特征
train_features = pd.get_dummies(train_features, columns=cat_features_M, dummy_na=True)
train_features = pd.get_dummies(train_features, columns=cat_features_L, dummy_na=False)
test_features = pd.get_dummies(test_features, columns=cat_features_M, dummy_na=True)
test_features = pd.get_dummies(test_features, columns=cat_features_L, dummy_na=False)


# 确保测试集的列与训练集对齐（缺少的补0），如果测试集有新类别，则这里认为新类别影响不大，直接忽略
test_features = test_features.reindex(columns=train_features.columns, fill_value=0)

# 把独热编码从bool型转换成float型
train_features = train_features.astype('float64')  
test_features = test_features.astype('float64')

# 转换成pytorch张量
feature_names = train_features.columns.to_numpy()  # 记录所有的特征名称，并将Pandas Index对象 转换为 numpy数组

train_features = torch.tensor(train_features.values, dtype=torch.float32)
test_features = torch.tensor(test_features.values, dtype=torch.float32)
train_labels = torch.tensor(train_labels.values.reshape(-1, 1), dtype=torch.float32)



#----------训练----------------
from torch import nn
from torch.utils import data 
import torch.nn.functional as F

# 输入特征的数量
in_features = train_features.shape[1]
# 定义模型(线性模型)
def get_net(): 
    net = nn.Sequential(nn.Linear(in_features, in_features),
                        nn.ReLU(),
                        nn.Linear(in_features, 1))
    return net 


# 使用弹性网络正则化作为损失函数，用于训练优化
class ElasticNetLoss(nn.Module):
    def __init__(self, regula_str=0.5, l1_ratio=0.5):
        super().__init__()
        self.regula_str = regula_str        # 控制正则化强度
        self.l1_ratio = l1_ratio  # 控制 L1 和 L2 正则化的比例
        
    def forward(self, pred, target, model):
        mse = F.mse_loss(pred, target)  # 计算均方误差 (MSE)
        l1_reg = 0.0
        l2_reg = 0.0
        # 避免对偏置项进行正则化
        for name, param in model.named_parameters():
            if 'weight' in name:
                l1_reg += torch.norm(param, 1)  # L1正则项：∑|w|
                l2_reg += torch.norm(param, 2) # L2正则项：∑w²
        # 结合 Elastic Net 损失
        return mse + self.regula_str * (self.l1_ratio * l1_reg + (1 - self.l1_ratio) * l2_reg)



# 考虑相对误差而不是绝对误差，因此使用logRMSE，只用于表示当前损失的程度，不用于优化
def log_rmse(net, features, labels):
    # 为了保持对数计算的稳定，将小于1e-6的数设置为1e-6
    preds = torch.clamp(net(features), min=1e-6)
    labels = torch.clamp(labels, min=1e-6)
    rmse = torch.sqrt(F.mse_loss(torch.log(preds), torch.log(labels)))
    return rmse.item() 

# 训练数据加载器(迭代器)
def load_data_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# 训练 
def train(net, train_features, train_labels, test_features, test_labels,
    num_epochs, learning_rate, regula_str, l1_ratio, batch_size):
    train_ls, test_ls = [], []
    train_iter = load_data_array((train_features, train_labels), batch_size)

    # Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # CyclicLR 学习率调度器
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 
                                                base_lr=learning_rate / 10,
                                                max_lr=learning_rate,
                                                step_size_up=20,
                                                mode="exp_range")
    loss = ElasticNetLoss(regula_str=regula_str, l1_ratio=l1_ratio)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y, net) # 计算损失，传入 net 作为参数以获取模型参数
            l.backward()
            optimizer.step()    # 更新模型参数
            scheduler.step()    # 更新学习率
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
def k_fold(k, X_train, y_train, num_epochs, learning_rate, regula_str, l1_ratio, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    train_l_aver, valid_l_aver = [0]*(num_epochs + 1), [0]*(num_epochs + 1)
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, 
                                   regula_str, l1_ratio, batch_size)

        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        train_l_aver = [x + y/k for x, y in zip(train_l_aver, train_ls)]
        valid_l_aver = [x + y/k for x, y in zip(valid_l_aver, valid_ls)]
        # if i == 0:
        #     # 画图，i=0训练损失和验证损失
        #     plt.figure() 
        #     plt.plot(list(range(1, num_epochs + 1)), train_ls, label='train', linestyle='-', color='blue')
        #     plt.plot(list(range(1, num_epochs + 1)), valid_ls, label='valid', linestyle='-', color='red')
        #     plt.yscale("log")
        #     plt.legend()
        #     plt.xlabel("epoch")
        #     plt.ylabel("log RMSE")
        #     plt.title('0-th data as valid_data')
        #     plt.grid(True)  
            
        print(f'{i + 1}th-fold, log rmse of train data: {float(train_ls[-1]):f}, '
              f'log rmse of valid data: {float(valid_ls[-1]):f}')

    plt.figure() 
    plt.plot(list(range(1, num_epochs + 1)), train_l_aver, label='train', linestyle='-', color='blue')
    plt.plot(list(range(1, num_epochs + 1)), valid_l_aver, label='valid', linestyle='-', color='red')
    plt.yscale("log")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("log RMSE")
    plt.title('average loss')
    plt.grid(True)   
    return train_l_aver[-1], valid_l_aver[-1] 


# ---------超参数选择，找到损失最小的一组超参数-----------
k, num_epochs, lr, regula_str, l1_ratio, batch_size = 5, 700, 0.1, 0.1, 0.1, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          regula_str, l1_ratio, batch_size)
print(f'{k}-fold: final average log rmse of train data: {float(train_l):f}'
      f'final average log rmse of valid data: {float(valid_l):f}')


# --------------保存模型参数---------------
# 用全部数据训练
net = get_net()
train_ls, _ = train(net, train_features, train_labels, None, None,
                    num_epochs, lr, regula_str, l1_ratio, batch_size)
# 画图，训练损失
plt.figure() 
plt.plot(np.arange(1, num_epochs + 1), train_ls, label='train')
plt.legend()
plt.yscale('log')
plt.xlabel('epoch')
plt.ylabel('log RMSE')
plt.grid(True)  # 显示网格 

print(f'Final log rmse of all train data: {float(train_ls[-1]):f}')

# 保存模型
torch.save(net.state_dict(), "model_house_preds.pth")
print("Training is complete! The model weights have saved.")


# -------------模型可解释性---------------
# 特征重要性分析
def feature_importance(net, features):
    """通过梯度分析特征重要性"""
    inputs = features.clone().requires_grad_(True)  # 克隆特征并启用梯度追踪
    outputs = net(inputs)  # 前向传播
    grads = torch.autograd.grad(outputs, inputs, 
                                grad_outputs=torch.ones_like(outputs))[0]  # 创建全1的梯度张量，并计算输出对输入的梯度
    return torch.mean(torch.abs(grads), dim=0)  # 取绝对值后求样本平均

# 随机抽取 100 个样本计算
generator = torch.Generator()  # 创建一个随机数生成器
generator.manual_seed(1)  # 设置随机种子
num_samples = 100
random_indices = torch.randperm(train_features.shape[0], generator=generator)[:num_samples]  # 生成随机排列的索引
imp = feature_importance(net, train_features[random_indices])

top10_val, top10_idx = torch.topk(imp, 10)  # 提取最大的10个值及其对应的索引
print("import features: ", feature_names[top10_idx])  # 打印特征名称

# 可视化
plt.figure(figsize=(10,6))
plt.barh(feature_names[top10_idx], imp[top10_idx])  # 水平条形图
plt.title('Gradient-based Feature Importance')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()  # 将最重要特征显示在顶部（默认的是在底部）
plt.tight_layout()  # 自动调整布局
plt.show()


# --------------预测---------------
# 将测试集应用于模型
preds = net(test_features).detach().numpy()
# 导出预测数据csv
test_data['SalePrice'] = pd.Series(preds.reshape(-1))
submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
submission.to_csv('submission.csv', index=False)


