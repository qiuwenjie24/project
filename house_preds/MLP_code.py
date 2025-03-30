
#-------------下载数据集到指定目录------------
import hashlib
import os
import requests

def download(name, cache_dir='./house_data'):
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

DATA_HUB['kaggle_house_train'] = (  
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  
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
# 定义多层感知机模型(MLP)
def get_net(): 
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net 


# 使用弹性网络正则化作为损失函数，用于训练优化
class ElasticNetLoss(nn.Module):
    def __init__(self, regula_str=0.0, l1_ratio=0.5):
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
def train(net, train_features, train_labels, valid_features, valid_labels,
    num_epochs, learning_rate, regula_str, l1_ratio, batch_size, is_optuna=False):
    train_ls, valid_ls = [], []
    train_iter = load_data_array((train_features, train_labels), batch_size)

    # Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)   #  实际学习率由调度器管理
    # CyclicLR 学习率调度器
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 
                                                base_lr=learning_rate / 400,    # 最低学习率
                                                max_lr=learning_rate,          # 最高学习率（初始学习率）
                                                step_size_up=1000,               # 上升所需的迭代步数
                                                mode="exp_range")              # 波动模式：指数衰减
    loss = ElasticNetLoss(regula_str=regula_str, l1_ratio=l1_ratio)
    
    # 早停参数
    patience = 10
    best_loss = float('inf')
    best_epoch = 0
    no_improve = 0

    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y, net) # 计算损失，传入 net 作为参数以获取模型参数
            l.backward()
            optimizer.step()    # 更新模型参数
            scheduler.step()    # 更新学习率

        train_ls.append(log_rmse(net, train_features, train_labels))
        if valid_labels is not None:
            current_val_loss = log_rmse(net, valid_features, valid_labels)
            valid_ls.append(current_val_loss)
            # 仅对验证集早停判断（但不进行提前终止，仅记录，因为后面需要每个回合的数据画图）
            if current_val_loss < best_loss:
                best_loss = current_val_loss
                best_epoch = epoch
                no_improve = 0
            elif is_optuna is True:   # 如果是超参数调优则使用早停
                no_improve += 1
                if no_improve >= patience:
                    break  # 提前终止训练

    return train_ls, valid_ls, best_epoch, best_loss  


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
def k_fold(k, X_train, y_train, num_epochs, learning_rate, regula_str, l1_ratio, batch_size, is_optuna=False):

    train_l_epochs, valid_l_epochs = [0]*(num_epochs + 1), [0]*(num_epochs + 1)
    all_best_epochs = []
    all_best_loss = []
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()   # 获取并初始化模型
        train_ls, valid_ls, best_epoch, best_loss = train(net, *data, num_epochs, learning_rate, 
                                   regula_str, l1_ratio, batch_size, is_optuna)

        all_best_epochs.append(best_epoch)
        all_best_loss.append(best_loss)
        if is_optuna is False:
            train_l_epochs = [x + y/k for x, y in zip(train_l_epochs, train_ls)]
            valid_l_epochs = [x + y/k for x, y in zip(valid_l_epochs, valid_ls)]
        

    return train_l_epochs, valid_l_epochs, all_best_epochs, all_best_loss  # 返回每个回合的损失 和 每折验证集的最佳回合、损失


# ---------超参数选择，使用自动调参找到损失最小的一组超参数-----------
import optuna
# Optuna 目标函数，定义单次超参数试验的完整流程，由 Optuna 自动调用。
def objective(trial):

    # 动态采样超参数
    k = trial.suggest_int("k", 5, 10)
    num_epochs = 300
    lr = trial.suggest_float("lr", 1e-1, 100, log=True)
    regula_str = trial.suggest_categorical("regula_str", [i*0.05 for i in range(0, 10, 1)] ) #trial.suggest_float("regula_str", 0, 0, log=True)
    l1_ratio = trial.suggest_categorical("l1_ratio", [i*0.1 for i in range(0, 5, 1)] ) #trial.suggest_float("l1_ratio", 0, 0)

    exponent = trial.suggest_int("batch_size_exponent", 5, 8) 
    batch_size =  2 ** exponent   # 2^4=16, 2^8=256

    _, _, all_best_epochs, all_best_loss = k_fold(k, train_features, train_labels, num_epochs, lr,
                                                    regula_str, l1_ratio, batch_size, is_optuna=True)

    final_best_epoch = int(np.mean(all_best_epochs))
    final_best_loss = np.mean(all_best_loss)
    # 将最佳轮数保存为 Trial 的用户属性
    trial.set_user_attr("best_epoch", final_best_epoch)
    trial.set_user_attr("max_epoch", num_epochs)
    return final_best_loss  # 返回优化目标（损失值）即平均最佳损失

# 创建 Study 用于记录所有试验的结果，并启动自动调参
# study = optuna.create_study(direction="minimize")  # 目标是最小化损失
# study.optimize(objective, n_trials=20) # 尝试 n_trials 组超参数组合

# 保存
import joblib
# joblib.dump(study, "MLP_study.pkl")
# 加载
study = joblib.load("MLP_study.pkl")

# 输出最佳结果
best_trial = study.best_trial  # 返回最佳试验，object对象
best_params = best_trial.params  # 返回最佳试验的超参数键值对
print(f'{best_params["k"]}-fold: final average log rmse of valid data: {float(best_trial.value):f}',  # 目标函数的返回值（即损失值）
      '\nbest parameters:', best_params, '\nbest epoch:', best_trial.user_attrs["best_epoch"])

# -----------当前参数下的验证损失图像---------
# 最优超参数
k = best_params["k"]
num_epochs = best_trial.user_attrs["best_epoch"]   # max_epoch, best_epoch
lr = best_params["lr"]
regula_str = best_params["regula_str"]
l1_ratio = best_params["l1_ratio"]
exponent = best_params["batch_size_exponent"] 
batch_size = 2 ** exponent   # 2^4=16, 2^8=256

train_l_epochs, valid_l_epochs, _, _ = k_fold(k, train_features, train_labels, num_epochs, lr,
                                        regula_str, l1_ratio, batch_size, is_optuna=False)
print(f'Final log rmse of valid data: {float(valid_l_epochs[-1]):f}')

# 画图，每个回合数的平均训练损失和平均验证损失
plt.figure() 
plt.plot(list(range(1, num_epochs + 1)), train_l_epochs, label='train', linestyle='-', color='blue')
plt.plot(list(range(1, num_epochs + 1)), valid_l_epochs, label='valid', linestyle='-', color='red')
plt.yscale("log")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("log RMSE")
plt.title('average loss---MLP')
plt.grid(True)
# plt.show()
# input()

# --------------保存模型参数---------------
# 用全部数据训练
best_epoch = best_trial.user_attrs["best_epoch"]

net = get_net()  # 初始化
train_ls, _, _, _ = train(net, train_features, train_labels, None, None,
                    best_epoch, lr, regula_str, l1_ratio, batch_size, is_optuna=False)
print(f'Final log rmse of all train data: {float(train_ls[-1]):f}')

# 画图，训练损失
plt.figure() 
plt.plot(np.arange(1, best_epoch + 1), train_ls, label='train')
plt.legend()
plt.yscale('log')
plt.xlabel('epoch')
plt.ylabel('log RMSE')
plt.grid(True)  # 显示网格 


# 保存模型
torch.save(net.state_dict(), "MLP_model.pth")
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
submission.to_csv('MLP_submission.csv', index=False)


