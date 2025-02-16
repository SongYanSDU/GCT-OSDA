import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Lambda, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.losses import BinaryCrossentropy
from main_models import (build_feature_extractor_bjut,  build_doc_model, build_domain_discriminator, GradientReversalLayer)
from data_loader import bjut_dataset, bjut_dataset_4
import os
from main_losses_bjut import total_loss, mmd_loss
from sklearn.preprocessing import LabelBinarizer
from main_classify_dynamic import classify_with_dynamic_thresholds
from scipy.stats import norm
import gc
import random as python_random
import scipy.io as sio
seed = 42
np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 禁用非错误级别的日志信息
num_classes = 4

def data_shuffle(data1, label1):
    index1 = np.arange(np.size(data1, 0))
    np.random.shuffle(index1)
    data1 = data1[index1, :]
    label1 = label1[index1, ]
    return data1, label1

class GaussianClassifier(Model):
    def __init__(self, num_classes, feat_dim):
        super(GaussianClassifier, self).__init__()
        # 定义 trainable 的 means 和 logvars
        self.means = tf.Variable(tf.random.normal([num_classes, feat_dim]), trainable=True)
        self.logvars = tf.Variable(tf.random.normal([num_classes, feat_dim]), trainable=True)

    def call(self, inputs):
        return self.means, self.logvars

def test_predict(combined_model, x_train_target, x_train_source, tt_label, gaussian_model):
    feats_target, cls_probs = combined_model.predict(x_train_target, verbose=0)  # (N, feat_dim), (N, 4)
    pred_label = np.argmax(cls_probs, axis=1)  # Predicted classes
    # 使用源域训练集计算高斯分布参数
    means_g = gaussian_model.means.numpy()
    stds_g = gaussian_model.logvars.numpy()
    # 对未知类别样本的处理
    results = classify_with_dynamic_thresholds(feats_target, means_g, stds_g, pred_label, z_score=2.0)
    results = results.numpy()
    # 推断阶段
    acc_p = np.mean(results == tt_label)
    pseudo_labels = results
    return feats_target, pred_label, acc_p, pseudo_labels

# 数据加载和准备
(data1_4, label1_4, data2_4, label2_4, data3_4, label3_4, data4_4, label4_4,
data5_4, label5_4, data6_4, label6_4, data7_4, label7_4, data8_4, label8_4) = bjut_dataset_4()
(data1, label1, data2, label2, data3, label3, data4, label4,
data5, label5, data6, label6, data7, label7, data8, label8) = bjut_dataset()
# 定义训练和测试对
# 组合源域数据
source_data_1 = np.concatenate((data1_4, data2_4), axis=0)
source_labels_1 = np.concatenate((label1_4, label2_4), axis=0)
source_data_2 = np.concatenate((data2_4, data3_4), axis=0)
source_labels_2 = np.concatenate((label2_4, label3_4), axis=0)
source_data_3 = np.concatenate((data3_4, data4_4), axis=0)
source_labels_3 = np.concatenate((label3_4, label4_4), axis=0)
source_data_4 = np.concatenate((data4_4, data5_4), axis=0)
source_labels_4 = np.concatenate((label4_4, label5_4), axis=0)
source_data_5 = np.concatenate((data5_4, data6_4), axis=0)
source_labels_5 = np.concatenate((label5_4, label6_4), axis=0)
source_data_6 = np.concatenate((data6_4, data7_4), axis=0)
source_labels_6 = np.concatenate((label6_4, label7_4), axis=0)

data_pairs = [(source_data_1, source_labels_1, data3, label3),
              (source_data_2, source_labels_2, data4, label4),
              (source_data_3, source_labels_3, data5, label5),
              (source_data_4, source_labels_4, data6, label6),
              (source_data_5, source_labels_5, data7, label7),
              (source_data_6, source_labels_6, data8, label8)]

for i, (tr_data, tr_label, tt_data, tt_label) in enumerate(data_pairs):
    print(f"******************************* data {i + 1} **************************************")
    tr_data, tr_label = data_shuffle(tr_data, tr_label)
    train_data = tr_data
    train_label = tr_label
    num = 4000
    tt_data_1 = tt_data[:num]
    tt_label_1 = tt_label[:num]
    tt_data_2 = tt_data[num:]
    tt_label_2 = tt_label[num:]
    # 数据标准化
    scaler = StandardScaler()
    combined_data = np.vstack([train_data.reshape(-1, 4096), tt_data.reshape(-1, 4096)])
    scaler.fit(combined_data)
    x_train_source = scaler.transform(train_data.reshape(-1, 4096)).reshape(-1, 4096, 1)
    x_train_target = scaler.transform(tt_data_1.reshape(-1, 4096)).reshape(-1, 4096, 1)
    tt_data_2 = scaler.transform(tt_data_2.reshape(-1, 4096)).reshape(-1, 4096, 1)
    # 初始化模型
    feature_extractor = build_feature_extractor_bjut()
    fault_diagnosis = build_doc_model(feature_extractor.output_shape[1], num_classes=num_classes)  # 4 known + 1 unknown
    gaussian_model = GaussianClassifier(num_classes, feature_extractor.output_shape[1])
    # combined model
    inp = Input(shape=(4096, 1))  # 特征输入
    feature_extractor_output = feature_extractor(inp)  # 得到 (batch, some_dim)
    feat_for_gaussian, cls_output = fault_diagnosis(feature_extractor_output)
    # Combined model
    lr_schedule = 3e-4
    optimizer = Adam(learning_rate=lr_schedule)
    combined_model = Model(inputs=inp, outputs=[feat_for_gaussian, cls_output])
    batch_size = 256

    num_samples_0 = len(x_train_source)
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(np.arange(num_classes))
    for epoch in range(300):
        _, _, _, pseudo_labels =  test_predict(combined_model, x_train_target, x_train_source, tt_label_1, gaussian_model)
        # pseudo_labels = pseudo_labels.numpy()
        known_mask = pseudo_labels < num_classes
        x_train_target_filtered = x_train_target[known_mask]
        x_train_target_label = pseudo_labels[known_mask]
        num_samples_1 = len(x_train_target_filtered)

        for step in range(len(x_train_source) // batch_size):
            # 获取当前批次数据
            indices_0 = np.random.randint(0, num_samples_0, size=batch_size)
            if num_samples_1 > batch_size:
                indices_1 = np.random.randint(0, num_samples_1, size=batch_size)
                target_batch = x_train_target_filtered[indices_1]
                target_labels = x_train_target_label[indices_1]
            else:
                target_batch = x_train_target_filtered
                target_labels = x_train_target_label
            source_batch = x_train_source[indices_0]
            source_labels = train_label[indices_0]

            with tf.GradientTape(persistent=True) as tape:
                source_features, source_cls_output = combined_model(source_batch)  # 源域前向传播
                target_features, target_cls_probs = combined_model(target_batch)   # 目标域前向传播
                source_labels_binary = label_binarizer.transform(source_labels)    # 分类任务损失
                classification_loss = tf.keras.losses.binary_crossentropy(label_binarizer.transform(source_labels),
                                                                          source_cls_output)
                means, logvars = gaussian_model(tf.stop_gradient(source_features))
                gaussian_loss = total_loss(source_labels, source_features, means, logvars, num_classes)
                m_loss = mmd_loss(source_features, target_features, sigma=1.0)
                sum_loss = classification_loss + gaussian_loss + m_loss

            # 分别更新 combined_model 和 gaussian_model
            combined_grads = tape.gradient(sum_loss, combined_model.trainable_variables)
            optimizer.apply_gradients(zip(combined_grads, combined_model.trainable_variables))

            gaussian_grads = tape.gradient(gaussian_loss, gaussian_model.trainable_variables)
            optimizer.apply_gradients(zip(gaussian_grads, gaussian_model.trainable_variables))
        # 每 50 个 epoch 调整学习率
        if (epoch + 1) % 20 == 0:
            current_lr = optimizer.lr.numpy()
            new_lr = current_lr * 0.8
            optimizer.lr.assign(new_lr)

        # ====== 目标域推断 ======
        final_means = gaussian_model.means.numpy()
        final_logvars = gaussian_model.logvars.numpy()
        # ======= 推断目标域并识别未知 =======
        _, _, acc_p, _ = test_predict(combined_model, tt_data_2, x_train_source, tt_label_2, gaussian_model)
        if (epoch + 1) % 20 == 0:
            print("gaussian_loss = ", tf.reduce_mean(gaussian_loss).numpy())
            print(f"Epoch: {epoch}, Accuracy_p: {acc_p}")  # 打印分类报告

    feats_target, _, _, pseudo_labels = test_predict(combined_model, tt_data_2[:2000], x_train_source, tt_label_2[:2000],
                                                     gaussian_model)
    source_features, _ = combined_model(x_train_source[:2000])
    final_means = gaussian_model.means.numpy()
    final_logvars = gaussian_model.logvars.numpy()
    # Create a directory to save results
    save_dir = "results_bjut_4"
    # Save results
    result_data = {
        "source_features": source_features,  # (N, 4096)
        "source_labels": train_label[:2000],  # (N,)
        "target_features": feats_target,  # (N, feat_dim)
        "target_labels": tt_label_2[:2000],  # (N,)
        "target_predict": pseudo_labels,
        "gaussian_means": final_means,  # (num_classes, feat_dim)
        "gaussian_logvars": final_logvars,  # (num_classes, feat_dim)
    }

    save_path = os.path.join(save_dir, f"experiment_{i + 1}.mat")
    sio.savemat(save_path, result_data)
    print(f"Saved experiment {i + 1} results to {save_path}")

    del combined_model
    del gaussian_model
    K.clear_session()
    # Optional: 如果使用GPU，建议强制释放显存，确保没有残留
    try:
        gc.collect()  # 强制进行垃圾回收
    except ImportError:
        pass

