import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from data_loader import bjut_dataset_4, bjut_dataset

# 构建特征提取网络
def build_feature_extractor(input_shape=(4096, 1)):
    inp = Input(shape=input_shape)
    x = Conv1D(16, kernel_size=33, activation='relu', padding='same')(inp)
    x = MaxPooling1D(pool_size=32)(x)
    x = Conv1D(32, kernel_size=9, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=8)(x)
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    features = Flatten()(x)
    return Model(inputs=inp, outputs=features)

# 构建 DOC 模型
def build_doc_model(feature_extractor, num_classes):
    inp = Input(shape=(4096, 1))
    features = feature_extractor(inp)
    features = Dense(128, activation='relu')(features)
    outputs = Dense(num_classes, activation='sigmoid')(features)  # One-vs-all sigmoid classifiers
    return Model(inputs=inp, outputs=outputs)

# 数据生成
# 数据加载和准备
(data1_4, label1_4, data2_4, label2_4, data3_4, label3_4, data4_4, label4_4,
 data5_4, label5_4, data6_4, label6_4, data7_4, label7_4, data8_4, label8_4) = bjut_dataset_4()
(data1, label1, data2, label2, data3, label3, data4, label4,
 data5, label5, data6, label6, data7, label7, data8, label8) = bjut_dataset()

# 数据准备
num_classes = 4
X, y = data1_4, label1_4
X_test, y_test = data3, label3

# 标签二值化
y_binary = LabelBinarizer().fit_transform(y)  # Convert to one-hot-like for sigmoid training

# 构建模型
feature_extractor = build_feature_extractor(input_shape=(4096, 1))
doc_model = build_doc_model(feature_extractor, num_classes=num_classes)
doc_model.compile(optimizer=Adam(3e-4), loss=BinaryCrossentropy(), metrics=['accuracy'])

for epoch in range(10):
    # 训练
    doc_model.fit(X, y_binary, batch_size=32, epochs=100, verbose=0)

    # 推断阶段
    pred_probs = doc_model.predict(X_test)
    threshold = 0.85  # Confidence threshold for known classes
    pred_labels = np.argmax(pred_probs, axis=1)  # Predicted classes
    unknown_mask = np.max(pred_probs, axis=1) < threshold  # Mark as unknown if all probabilities < threshold
    pred_labels[unknown_mask] = num_classes  # Assign to unknown class

    # 打印分类报告
    # print(classification_report(y_test, pred_labels, target_names=[f"Class {i}" for i in range(num_classes)] + ["Unknown"]))
    acc = np.mean(pred_labels == y_test)
    print(f"Epoch: {epoch}, Accuracy: {acc}")
