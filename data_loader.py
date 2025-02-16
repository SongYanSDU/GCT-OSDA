# -- coding: utf-8 --
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler

def data_shuffle(data1, label1):
    index1 = np.arange(np.size(data1, 0))
    np.random.shuffle(index1)
    data1 = data1[index1, :]
    label1 = label1[index1, ]
    return data1, label1

def Lboro_dataset_6():
    def load_and_filter(filepath):
        pd = sio.loadmat(filepath)
        data = pd[list(pd.keys())[-1]][:, 1:]  # 通常动态地获取变量名
        label = pd[list(pd.keys())[-1]][:, 0]
        return data_shuffle(data, label)

    data1, label1 = load_and_filter('F:\\数据集\\拉夫堡\\lboro_900.mat')
    data2, label2 = load_and_filter('F:\\数据集\\拉夫堡\\lboro_1500.mat')
    data3, label3 = load_and_filter('F:\\数据集\\拉夫堡\\lboro_2000.mat')
    data4, label4 = load_and_filter('F:\\数据集\\拉夫堡\\lboro_1500_2000.mat')
    data5, label5 = load_and_filter('F:\\数据集\\拉夫堡\\lboro_900_1500.mat')
    data6, label6 = load_and_filter('F:\\数据集\\拉夫堡\\lboro_900_2000.mat')
    return data1, label1, data2, label2, data3, label3, data4, label4, data5, label5, data6, label6

def Lboro_dataset_5():
    def load_and_filter(filepath):
        pd = sio.loadmat(filepath)
        data = pd[list(pd.keys())[-1]][:, 1:]  # 通常动态地获取变量名
        label = pd[list(pd.keys())[-1]][:, 0]
        # 只保留标签为0到4的数据
        mask = (label >= 0) & (label <= 4)
        data, label = data[mask], label[mask]
        return data_shuffle(data, label)

    data1, label1 = load_and_filter('F:\\数据集\\拉夫堡\\lboro_900.mat')
    data2, label2 = load_and_filter('F:\\数据集\\拉夫堡\\lboro_1500.mat')
    data3, label3 = load_and_filter('F:\\数据集\\拉夫堡\\lboro_2000.mat')
    data4, label4 = load_and_filter('F:\\数据集\\拉夫堡\\lboro_1500_2000.mat')
    data5, label5 = load_and_filter('F:\\数据集\\拉夫堡\\lboro_900_1500.mat')
    data6, label6 = load_and_filter('F:\\数据集\\拉夫堡\\lboro_900_2000.mat')
    return data1, label1, data2, label2, data3, label3, data4, label4, data5, label5, data6, label6

def Lboro_dataset_4():
    def load_and_filter(filepath):
        pd = sio.loadmat(filepath)
        data = pd[list(pd.keys())[-1]][:, 1:]  # 通常动态地获取变量名
        label = pd[list(pd.keys())[-1]][:, 0]
        # 只保留标签为0到4的数据
        mask = (label >= 0) & (label <= 3)
        data, label = data[mask], label[mask]
        return data_shuffle(data, label)

    data1, label1 = load_and_filter('F:\\数据集\\拉夫堡\\lboro_900.mat')
    data2, label2 = load_and_filter('F:\\数据集\\拉夫堡\\lboro_1500.mat')
    data3, label3 = load_and_filter('F:\\数据集\\拉夫堡\\lboro_2000.mat')
    data4, label4 = load_and_filter('F:\\数据集\\拉夫堡\\lboro_1500_2000.mat')
    data5, label5 = load_and_filter('F:\\数据集\\拉夫堡\\lboro_900_1500.mat')
    data6, label6 = load_and_filter('F:\\数据集\\拉夫堡\\lboro_900_2000.mat')
    return data1, label1, data2, label2, data3, label3, data4, label4, data5, label5, data6, label6

def bjut_dataset():
    def load_and_filter(filepath):
        pd = sio.loadmat(filepath)
        data = pd[list(pd.keys())[-1]][:, 1:]  # 通常动态地获取变量名
        label = pd[list(pd.keys())[-1]][:, 0]
        unique_classes = np.unique(label)
        selected_data, selected_label = [], []
        sample_per_class = 2000
        for cls in unique_classes:
            idx = np.where(label == cls)[0]  # 找到该类别的索引
            np.random.shuffle(idx)  # 打乱索引
            selected_idx = idx[:sample_per_class]  # 选取前 sample_per_class 个样本
            selected_data.append(data[selected_idx])
            selected_label.append(label[selected_idx])
        return data_shuffle(np.vstack(selected_data), np.hstack(selected_label))

    data1, label1 = load_and_filter('F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_20.mat')
    data2, label2 = load_and_filter('F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_25.mat')
    data3, label3 = load_and_filter('F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_30.mat')
    data4, label4 = load_and_filter('F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_35.mat')
    data5, label5 = load_and_filter('F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_40.mat')
    data6, label6 = load_and_filter('F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_45.mat')
    data7, label7 = load_and_filter('F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_50.mat')
    data8, label8 = load_and_filter('F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_55.mat')
    return (data1, label1, data2, label2, data3, label3, data4, label4, data5, label5,
            data6, label6, data7, label7, data8, label8)

def bjut_dataset_4():
    def load_and_filter(filepath):
        pd = sio.loadmat(filepath)
        data = pd[list(pd.keys())[-1]][:, 1:]  # 通常动态地获取变量名
        label = pd[list(pd.keys())[-1]][:, 0]
        mask = (label >= 0) & (label <= 3)
        data, label = data[mask], label[mask]
        unique_classes = np.unique(label)
        selected_data, selected_label = [], []
        sample_per_class = 2000
        for cls in unique_classes:
            idx = np.where(label == cls)[0]  # 找到该类别的索引
            np.random.shuffle(idx)  # 打乱索引
            selected_idx = idx[:sample_per_class]  # 选取前 sample_per_class 个样本
            selected_data.append(data[selected_idx])
            selected_label.append(label[selected_idx])
        return data_shuffle(np.vstack(selected_data), np.hstack(selected_label))

    data1, label1 = load_and_filter('F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_20.mat')
    data2, label2 = load_and_filter('F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_25.mat')
    data3, label3 = load_and_filter('F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_30.mat')
    data4, label4 = load_and_filter('F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_35.mat')
    data5, label5 = load_and_filter('F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_40.mat')
    data6, label6 = load_and_filter('F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_45.mat')
    data7, label7 = load_and_filter('F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_50.mat')
    data8, label8 = load_and_filter('F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_55.mat')
    return (data1, label1, data2, label2, data3, label3, data4, label4, data5, label5,
            data6, label6, data7, label7, data8, label8)

def bjut_dataset_3():
    def load_and_filter(filepath):
        pd = sio.loadmat(filepath)
        data = pd[list(pd.keys())[-1]][:, 1:]  # 通常动态地获取变量名
        label = pd[list(pd.keys())[-1]][:, 0]
        # 只保留标签为0到4的数据
        mask = (label >= 0) & (label <= 2)
        data, label = data[mask], label[mask]
        unique_classes = np.unique(label)
        selected_data, selected_label = [], []
        sample_per_class = 2000
        for cls in unique_classes:
            idx = np.where(label == cls)[0]  # 找到该类别的索引
            np.random.shuffle(idx)  # 打乱索引
            selected_idx = idx[:sample_per_class]  # 选取前 sample_per_class 个样本
            selected_data.append(data[selected_idx])
            selected_label.append(label[selected_idx])
        return data_shuffle(np.vstack(selected_data), np.hstack(selected_label))

    data1, label1 = load_and_filter('F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_20.mat')
    data2, label2 = load_and_filter('F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_25.mat')
    data3, label3 = load_and_filter('F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_30.mat')
    data4, label4 = load_and_filter('F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_35.mat')
    data5, label5 = load_and_filter('F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_40.mat')
    data6, label6 = load_and_filter('F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_45.mat')
    data7, label7 = load_and_filter('F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_50.mat')
    data8, label8 = load_and_filter('F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_55.mat')
    return (data1, label1, data2, label2, data3, label3, data4, label4, data5, label5,
            data6, label6, data7, label7, data8, label8)

