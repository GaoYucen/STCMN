#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
import copy
import re

#%% 读入.txt文件
f = open('../data/traffic_flow/1/20230306/part-00000', 'r')
inline = f.readlines()
# 创建含'id'和'flow'关键字的dict
data = []
for line in inline:
    data_line = re.split('[ ;]', line.strip("\n"))
    # 根据空格和分号划分line
    data.append({'id': int(data_line[0]), 'flow': np.array([float(i.split(',')[5]) for i in data_line[1:]])})

#%% 确定data['flow']长度为288的id列表
id_list = [d['id'] for d in data if len(d['flow']) == 288]

# 保留data['id']在id_list中的条目
data1 = [d for d in data if d['id'] in id_list]

# 将data1中的'flow'抽取为新的numpy数组
data_flow = np.array([d['flow'] for d in data1])

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

#%% 对data['flow']进行归一化
data = pd.DataFrame(data)
max_value = np.max(data_flow)
data_flow = data_flow / max_value

#%% # Split data_flow into training and testing sets
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
    train_y.append(np.array(train_data[i+24:i+24+6, :]))
train_x = np.array(train_x)
train_y = np.array(train_y).reshape(-1, num_nodes * 6)

test_x, test_y = [], []

for i in range(time_len - train_size - 24 - 6):
    test_x.append(np.array(test_data[i:i+24, :]))
    test_y.append(np.array(test_data[i+24:i+24+6, :]))
test_x = np.array(test_x)
test_y = np.array(test_y).reshape(-1, num_nodes * 6)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_x = torch.from_numpy(train_x).float().to(device)
train_y = torch.from_numpy(train_y).float().to(device)
test_x = torch.from_numpy(test_x).float().to(device)
test_y = torch.from_numpy(test_y).float().to(device)

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Instantiate the LSTM model
input_size = num_nodes
hidden_size = 64
output_size = num_nodes * 6
num_layers = 2

model = LSTM(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss()

#%% training
optimizer = optim.Adam(model.parameters(), lr=0.01)

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

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'STCMN/param/no/checkpoint_mul.pth')  # 这里会存储迄今最优模型的参数
        # torch.save(model, 'finish_model.pkl') # 这里会存储迄今最优的模型
        self.val_loss_min = val_loss

early_stopping = EarlyStopping(patience=10, verbose=True)

for epoch in range(10000):
    model.train()
    optimizer.zero_grad()
    output = model(train_x)
    loss = criterion(output, train_y)
    loss.backward()
    optimizer.step()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 10000, loss.item()))
    early_stopping(loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

#%% 读取模型参数
model.load_state_dict(torch.load('STCMN/param/no/checkpoint_mul.pth'))

#%% testing
model.eval()
with torch.no_grad():
    output = model(test_x)
    loss = criterion(output, test_y)
    print('Test Loss: {:.4f}'.format(loss.item()))

#%% 转化数据
output = output.cpu().numpy()
test_y = test_y.cpu().numpy()

#%% 针对测试集输出RMSE结果
test_y = test_y * max_value
output = output * max_value

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

#%% # 绘制test_y_true[:,0:25]和output_true[:,0:25]的对比折线图
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(test_y[:,0:25].flatten(), label='true')
plt.plot(output[:,0:25].flatten(), label='predicted')
plt.legend()
plt.show()

#%% Plot the predicted and true values
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(test_y.flatten()[1:100], label='true')
plt.plot(output.flatten()[1:100], label='predicted')
plt.legend()
plt.show()
