import pandas as pd
import numpy as np
import re

#%% 读入.txt文件
f = open('data/traffic_flow/1/20230306/part-00000', 'r')
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

# 共7995个拥有完整288个流量数据的link（esiwei数据，包括方向）
# id_list标记link_id
# data_flow标记对应的traffic flow

#%% 读取features信息
feature = pd.read_csv('data/link_feature.txt', sep='\t')
# 共80700个link，每个link有53个特征（siwei数据，不包括方向）

# 确定割边的link_id列表
id_list_cut_total = []
# 依次取feature中的元素
for i in range(len(feature)):
    # 如果该元素的partition为1，则将其link_ID添加到id_list_cut中
    if feature['partition'][i] == 1:
        id_list_cut_total.append(feature['link_ID'][i])

# 从idlist中找到割边对应的idlist，以及对应的flow
# 从id_list中分离出cut和no_cut
id_list_cut = [i for i in id_list if i/10 in id_list_cut_total]
id_list_no_cut = [i for i in id_list if i/10 not in id_list_cut_total]

# 从data_flow中分离出cut和no_cut
data_flow_cut = data_flow[[i for i,x in enumerate(id_list) if x in id_list_cut]]
data_flow_no_cut = data_flow[[i for i,x in enumerate(id_list) if x in id_list_no_cut]]

# 共7749个非割边，246个割边

# #%% 从原data中找到割边对应的idlist，以及对应的flow
# id_list_cut = [d['id'] for d in data if d['id']/10 in id_list_cut_total]
# data_flow_cut = [d['flow'] for d in data if d['id'] in id_list_cut]

#%% 对比data_flow_cut和data_flow_no_cut的区别
# 从data_flow_cut中采样5条边，并绘图
import matplotlib.pyplot as plt
import random
k = 3
# 从data_flow_cut中随机选取k条边
id_list_cut_sample = random.sample(id_list_cut, k)
# 从data_flow_cut中选取对应的flow
data_flow_cut_sample = data_flow_cut[[i for i,x in enumerate(id_list_cut) if x in id_list_cut_sample]]
# 绘图
plt.figure(figsize=(10, 5))
for i in range(len(data_flow_cut_sample)):
    plt.plot(data_flow_cut_sample[i], linewidth =3.0, label='cut ' + str(i))
plt.legend()

# 从data_flow_no_cut中采样5条边，并绘图
# 从data_flow_no_cut中随机选取5条边
id_list_no_cut_sample = random.sample(id_list_no_cut, k)
# 从data_flow_no_cut中选取对应的flow
data_flow_no_cut_sample = data_flow_no_cut[[i for i,x in enumerate(id_list_no_cut) if x in id_list_no_cut_sample]]
# 绘图
for i in range(len(data_flow_no_cut_sample)):
    plt.plot(data_flow_no_cut_sample[i], label='nocut ' + str(i))
plt.legend()
plt.savefig('graph/cut_vs_nocut.pdf')

#%% 统计cut和nocut流量的统计特征
# 1. 找到data_flow_cut和data_flow_no_cut的均值
print(np.mean(data_flow_cut))
print(np.mean(data_flow_no_cut))
# 2. 方差
print(np.var(data_flow_cut))
print(np.var(data_flow_no_cut))
# # 3. 偏度
# print(pd.DataFrame(data_flow_cut).skew())
# print(pd.DataFrame(data_flow_no_cut).skew())
# # 4. 峰度
# print(pd.DataFrame(data_flow_cut).kurt())
# print(pd.DataFrame(data_flow_no_cut).kurt())
# 5. 最大值
print(np.max(data_flow_cut))
print(np.max(data_flow_no_cut))
# 6. 最小值
print(np.min(data_flow_cut))
print(np.min(data_flow_no_cut))
# 7. 最大值位置
print(np.argmax(data_flow_cut)%288)
print(np.argmax(data_flow_no_cut)%288)
# 8. 最小值位置
print(np.argmin(data_flow_cut)%288)
print(np.argmin(data_flow_no_cut)%288)





