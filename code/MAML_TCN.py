#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
import copy
import re
import time

#%%
def load_data(filename):
    f = open(filename, 'r')
    inline = f.readlines()
    # 创建含'id'和'flow'关键字的dict
    data = []
    for line in inline:
        data_line = re.split('[ ;]', line.strip("\n"))
        # 根据空格和分号划分line
        data.append({'id': int(data_line[0]), 'flow': np.array([float(i.split(',')[5]) for i in data_line[1:]])})

    # 确定data['flow']长度为288的id列表
    id_list = [d['id'] for d in data if len(d['flow']) == 288]

    # 保留data['id']在id_list中的条目
    data1 = [d for d in data if d['id'] in id_list]

    # 将data1中的'flow'抽取为新的numpy数组
    data_flow = np.array([d['flow'] for d in data1])

    return data_flow, id_list

#%% 加载数据并准备训练
data_flow, id_list = load_data('../data/traffic_flow/1/20230306/part-00000')

#%% 读取features信息
feature = pd.read_csv('../data/punch_partition/output_2.txt', sep='\t')

# 将feature中的link_ID后添加0和1作为两行数据，形成新的feature
feature_0 = feature.copy()
feature_0['link_ID'] = feature_0['link_ID'] * 10
feature_1 = feature.copy()
feature_1['link_ID'] = feature_1['link_ID'] * 10 + 1
feature = pd.concat([feature_0, feature_1])

#%% 确定节点数和时间步数量
# 对data['flow']进行归一化
max_value = np.max(data_flow)
data_flow = data_flow / max_value

# 删除id_list不在feature['link_ID']中的条目，并删除data_flow对应的row
data_flow = data_flow[[i for i,x in enumerate(id_list) if x in feature['link_ID'].values]]
id_list = [i for i in id_list if i in feature['link_ID'].values]

time_len = data_flow.shape[1]
num_nodes = data_flow.shape[0]

# plit data_flow into training and testing sets
train_ratio = 0.8
train_size = int(time_len * train_ratio)
train_data = data_flow[:, :train_size]
test_data = data_flow[:, train_size:]
# Transpose train_data and test_data
train_data = np.transpose(train_data, (1, 0))
test_data = np.transpose(test_data, (1, 0))

#%%# Split train_data into train_x and train_y
# Use 24 steps to predict the next 6 steps
train_x, train_y = [], []

for i in range(train_size - 24 - 6):
    train_x.append(np.array(train_data[i:i+24, :]))
    train_y.append(np.array(train_data[i + 24:i + 24 + 6, :]))
train_x = np.transpose(np.array(train_x), (0,2,1)).reshape(-1,24)
train_y = np.transpose(np.array(train_y), (0,2,1)).reshape(-1,6)

test_x, test_y = [], []

for i in range(time_len - train_size - 24 - 6):
    test_x.append(np.array(test_data[i:i+24, :]))
    test_y.append(np.array(test_data[i + 24:i + 24 + 6, :]))
test_x = np.transpose(np.array(test_x), (0,2,1)).reshape(-1,24)
test_y = np.transpose(np.array(test_y), (0,2,1)).reshape(-1,6)

#%% 根据id_list筛选feature['link_ID']对应的row组成新的feature_road
feature_road = feature[feature['link_ID'].isin(id_list)]

# 将feature_road按照id_list的顺序进行排序
feature_road = feature_road.set_index('link_ID')
feature_road = feature_road.loc[id_list]
feature_road = feature_road.reset_index()

# feature去掉link_ID列和geometry列
feature_used = feature_road.drop(columns=['link_ID', 'geometry', 'Kind'])

# Drop rows with NaN values in feature_used
feature_used = feature_used.dropna(axis=1)

# Define a function to implement K-Means algorithm
def kmeans(data, k):
    """
    :param data: 2D numpy array
    :param k: number of clusters
    :return: labels of each data point
    """
    # Initialize centroids randomly
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    # Initialize labels as zeros
    labels = np.zeros(data.shape[0])
    # Iterate until convergence
    while True:
        # Calculate distances between each data point and each centroid
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        # Assign each data point to the closest centroid
        new_labels = np.argmin(distances, axis=0)
        # Check if labels have changed
        if np.array_equal(new_labels, labels):
            break
        # Update labels and centroids
        labels = new_labels
        for i in range(k):
            centroids[i] = np.mean(data[labels == i], axis=0)
    return labels

# Use kmeans function to cluster feature_used into 3 clusters
feature_used = np.array(feature_used.astype('float64'))
# 对feature_used的每一列进行归一化
for i in range(feature_used.shape[1]):
    max_value_1 = np.max(feature_used[:, i])
    feature_used[:, i] = feature_used[:, i] / max_value_1
# 删除含nan的列
feature_used = feature_used[:, ~np.isnan(feature_used).any(axis=0)]
# 对data_flow的每一行的每12列进行求和平均
data_flow_new = np.zeros((data_flow.shape[0], 12))
for i in range(data_flow.shape[0]):
    for j in range(12):
        data_flow_new[i, j] = np.mean(data_flow[i, j*24:(j+1)*24])
# 将feature_uesd和data_flow进行拼接
feature_used_new = np.concatenate((feature_used, data_flow_new), axis=1)
feature_used_label = kmeans(feature_used_new, 3)
#%% 存储feature_used_label到.txt文件
np.savetxt('STCMN/TCN/param/all/feature_used_label.txt', feature_used_label)

#%% 读取feature_used_label
feature_used_label = np.loadtxt('STCMN/TCN/param/all/feature_used_label.txt')
# 统计各项出现的次数
feature_used_label_count = np.zeros(3)
for i in range(feature_used_label.shape[0]):
    feature_used_label_count[int(feature_used_label[i])] += 1
print(feature_used_label_count)

#%% Split train_x and train_y into three parts according to feature_used_label
train_x_0, train_x_1, train_x_2 = [], [], []
train_y_0, train_y_1, train_y_2 = [], [], []

for i in range(feature_used_label.shape[0]):
    if feature_used_label[i] == 0:
        train_x_0.append(train_x[(train_size-24-6)*i:(train_size-24-6)*(i+1),:])
        train_y_0.append(train_y[(train_size-24-6)*i:(train_size-24-6)*(i+1),:])
    elif feature_used_label[i] == 1:
        train_x_1.append(train_x[(train_size-24-6)*i:(train_size-24-6)*(i+1),:])
        train_y_1.append(train_y[(train_size-24-6)*i:(train_size-24-6)*(i+1),:])
    elif feature_used_label[i] == 2:
        train_x_2.append(train_x[(train_size-24-6)*i:(train_size-24-6)*(i+1),:])
        train_y_2.append(train_y[(train_size-24-6)*i:(train_size-24-6)*(i+1),:])

#%%
train_x_0 = np.array(train_x_0).reshape(-1,24)
train_x_1 = np.array(train_x_1).reshape(-1,24)
train_x_2 = np.array(train_x_2).reshape(-1,24)
train_y_0 = np.array(train_y_0).reshape(-1,6)
train_y_1 = np.array(train_y_1).reshape(-1,6)
train_y_2 = np.array(train_y_2).reshape(-1,6)

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_x_0 = torch.from_numpy(train_x_0).float().unsqueeze(2).to(device)
train_y_0 = torch.from_numpy(train_y_0).float().to(device)
train_x_1 = torch.from_numpy(train_x_1).float().unsqueeze(2).to(device)
train_y_1 = torch.from_numpy(train_y_1).float().to(device)
train_x_2 = torch.from_numpy(train_x_2).float().unsqueeze(2).to(device)
train_y_2 = torch.from_numpy(train_y_2).float().to(device)

#%% Define the TCN model
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, num_channels, kernel_size, stride=1, padding=(kernel_size-1)//2),
            nn.BatchNorm1d(num_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, num_channels, kernel_size, stride=1, padding=(kernel_size-1)//2),
            nn.BatchNorm1d(num_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, num_channels, kernel_size, stride=1, padding=(kernel_size-1)//2),
            nn.BatchNorm1d(num_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, num_channels, kernel_size, stride=1, padding=(kernel_size-1)//2),
            nn.BatchNorm1d(num_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, output_size, kernel_size, stride=1, padding=(kernel_size-1)//2)
        )

    def forward(self, inputs):
        # inputs shape: (batch_size, input_size, seq_len)
        y = self.tcn(inputs)
        # y shape: (batch_size, output_size, seq_len)
        return y[:, :, -1]

# Set hyperparameters
input_size = 24
output_size = 6
num_channels = 64
kernel_size = 3
dropout = 0.2
lr_set = 0.01
num_epochs = 50

# Initialize model, loss function, and optimizer
model0 = TCN(input_size, output_size, num_channels, kernel_size, dropout).to(device)
model1 = TCN(input_size, output_size, num_channels, kernel_size, dropout).to(device)
model2 = TCN(input_size, output_size, num_channels, kernel_size, dropout).to(device)
criterion = nn.MSELoss()

#%% training
optimizer0 = optim.Adam(model0.parameters(), lr=lr_set)
optimizer1 = optim.Adam(model1.parameters(), lr=lr_set)
optimizer2 = optim.Adam(model2.parameters(), lr=lr_set)

# Train the model until convergence
# Stop training when the validation loss stops decreasing for a certain number of epochs (e.g. 10) using EarlyStopping

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, task_kind):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, task_kind)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, task_kind)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, task_kind):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'STCMN/TCN/param/all/checkpoint_mul_'+str(task_kind)+'.pth')  # 这里会存储迄今最优模型的参数
        # torch.save(model, 'finish_model.pkl') # 这里会存储迄今最优的模型
        self.val_loss_min = val_loss

# Record the start time of training
start_time = time.time()

early_stopping_0 = EarlyStopping(patience=10, verbose=True)

for epoch in range(num_epochs):
    model0.train()
    optimizer0.zero_grad()
    output = model0(train_x_0)
    loss = criterion(output, train_y_0)
    loss.backward()
    optimizer0.step()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    early_stopping_0(loss, model0, 0)
    if early_stopping_0.early_stop:
        print("Early stopping")
        break

early_stopping_1 = EarlyStopping(patience=10, verbose=True)

for epoch in range(num_epochs):
    model1.train()
    optimizer1.zero_grad()
    output = model1(train_x_1)
    loss = criterion(output, train_y_1)
    loss.backward()
    optimizer1.step()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    early_stopping_1(loss, model1, 1)
    if early_stopping_1.early_stop:
        print("Early stopping")
        break

early_stopping_2 = EarlyStopping(patience=10, verbose=True)

for epoch in range(num_epochs):
    model2.train()
    optimizer2.zero_grad()
    output = model2(train_x_2)
    loss = criterion(output, train_y_2)
    loss.backward()
    optimizer2.step()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    early_stopping_2(loss, model2, 2)
    if early_stopping_2.early_stop:
        print("Early stopping")
        break

# Record the end time of training and calculate the total training time
end_time = time.time()
training_time = end_time - start_time
print("Total training time: {:.2f} seconds".format(training_time))

#%% 读取模型参数
model0.load_state_dict(torch.load('STCMN/TCN/param/all/checkpoint_mul_0.pth'))
model1.load_state_dict(torch.load('STCMN/TCN/param/all/checkpoint_mul_1.pth'))
model2.load_state_dict(torch.load('STCMN/TCN/param/all/checkpoint_mul_2.pth'))


#%% Split test_x and test_y into three parts according to feature_used_label
test_x_0, test_x_1, test_x_2 = [], [], []
test_y_0, test_y_1, test_y_2 = [], [], []

for i in range(feature_used_label.shape[0]):
    if feature_used_label[i] == 0:
        test_x_0.append(test_x[(time_len - train_size - 24 - 6)*i:(time_len - train_size - 24 - 6)*(i+1),:])
        test_y_0.append(test_y[(time_len - train_size - 24 - 6)*i:(time_len - train_size - 24 - 6)*(i+1),:])
    elif feature_used_label[i] == 1:
        test_x_1.append(test_x[(time_len - train_size - 24 - 6)*i:(time_len - train_size - 24 - 6)*(i+1),:])
        test_y_1.append(test_y[(time_len - train_size - 24 - 6)*i:(time_len - train_size - 24 - 6)*(i+1),:])
    else:
        test_x_2.append(test_x[(time_len - train_size - 24 - 6)*i:(time_len - train_size - 24 - 6)*(i+1),:])
        test_y_2.append(test_y[(time_len - train_size - 24 - 6)*i:(time_len - train_size - 24 - 6)*(i+1),:])

#%%
test_x_0 = np.array(test_x_0).reshape(-1,24)
test_x_1 = np.array(test_x_1).reshape(-1,24)
test_x_2 = np.array(test_x_2).reshape(-1,24)
test_y_0 = np.array(test_y_0).reshape(-1,6)
test_y_1 = np.array(test_y_1).reshape(-1,6)
test_y_2 = np.array(test_y_2).reshape(-1,6)

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_x_0 = torch.from_numpy(test_x_0).float().unsqueeze(2).to(device)
test_y_0 = torch.from_numpy(test_y_0).float().to(device)
test_x_1 = torch.from_numpy(test_x_1).float().unsqueeze(2).to(device)
test_y_1 = torch.from_numpy(test_y_1).float().to(device)
test_x_2 = torch.from_numpy(test_x_2).float().unsqueeze(2).to(device)
test_y_2 = torch.from_numpy(test_y_2).float().to(device)

#%% testing
model0.eval()
with torch.no_grad():
    output_0 = model0(test_x_0)
    loss = criterion(output_0, test_y_0)
    print('Test Loss: {:.4f}'.format(loss.item()))

model1.eval()
with torch.no_grad():
    output_1 = model1(test_x_1)
    loss = criterion(output_1, test_y_1)
    print('Test Loss: {:.4f}'.format(loss.item()))

model2.eval()
with torch.no_grad():
    output_2 = model2(test_x_2)
    loss = criterion(output_2, test_y_2)
    print('Test Loss: {:.4f}'.format(loss.item()))

#%% 转化数据
output_0 = output_0.cpu().numpy()
test_y_0 = test_y_0.cpu().numpy()
output_1 = output_1.cpu().numpy()
test_y_1 = test_y_1.cpu().numpy()
output_2 = output_2.cpu().numpy()
test_y_2 = test_y_2.cpu().numpy()

#%% 拼接output_0, output_1, output_2
output = np.concatenate((output_0, output_1, output_2), axis=0)
test_y = np.concatenate((test_y_0, test_y_1, test_y_2), axis=0)

#%% 针对测试集输出RMSE结果
test_y = test_y * 97.2
output = output * 97.2

#%%
# 针对测试集输出MSE，MAE，RMSE，MAPE，R2结果
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import check_array

# Calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true.flatten()), np.array(y_pred.flatten())
    # Delete elements in y_pred that correspond to the deleted elements in y_true
    y_pred = np.delete(y_pred, np.where(y_true == 0))
    # Delete elements in y_true that are equal to 0
    y_true = y_true[y_true != 0]
    return np.mean(np.abs((y_true - y_pred) / y_true))

mse = mean_squared_error(test_y, output)
mae = mean_absolute_error(test_y, output)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(test_y, output)
r2 = r2_score(test_y.flatten(), output.flatten())
print('Test MSE: {:.4f}'.format(mse))
print('Test MAE: {:.4f}'.format(mae))
print('Test RMSE: {:.4f}'.format(rmse))
print('Test MAPE: {:.4f}%'.format(mape))
print('Test R2: {:.4f}'.format(r2))

#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(test_y.flatten()[0:90] / 4, label='True')
plt.plot(output.flatten()[0:90] / 4, label='Predicted')
plt.legend()
# 添加y轴标签为m/s
plt.xlabel('Time Point (5min)')
plt.ylabel('Traffic Flow (m/s)')
plt.savefig('STCMN/graph/MAML_predict.pdf')

#%%
data_flow = data_flow * 97.2
print(np.mean(data_flow))