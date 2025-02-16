import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Lambda, concatenate, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf
from keras import backend as K
from tensorflow.python.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D

from data_loader import bjut_dataset_4, bjut_dataset
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 禁用非错误级别的日志信息
import random as python_random
seed = 42
np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)


def compute_mmd_loss(source_features, target_features, kernel='rbf', bandwidth=1.0):
    def rbf_kernel(x, y, bandwidth):
        """RBF 核"""
        xx = tf.reduce_sum(x ** 2, axis=1, keepdims=True)
        yy = tf.reduce_sum(y ** 2, axis=1, keepdims=True)
        xy = tf.matmul(x, y, transpose_b=True)
        dist = xx - 2 * xy + tf.transpose(yy)
        return tf.exp(-dist / (2.0 * bandwidth ** 2))

    if kernel == 'rbf':
        # 计算 RBF 核矩阵
        K_ss = rbf_kernel(source_features, source_features, bandwidth)
        K_tt = rbf_kernel(target_features, target_features, bandwidth)
        K_st = rbf_kernel(source_features, target_features, bandwidth)

        # MMD 损失计算
        mmd_loss = tf.reduce_mean(K_ss) + tf.reduce_mean(K_tt) - 2 * tf.reduce_mean(K_st)
    else:
        raise ValueError(f"Unsupported kernel type: {kernel}")

    return mmd_loss


class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self, lambda_weight=1.0, **kwargs):
        super(GradientReversalLayer, self).__init__(**kwargs)
        self.lambda_weight = lambda_weight

    def call(self, x):
        @tf.custom_gradient
        def grad_reverse(x):
            def custom_grad(dy):
                return -self.lambda_weight * dy
            return x, custom_grad
        return grad_reverse(x)


# 特征提取网络
def build_feature_extractor_bjut():
    inputs = Input(shape=(4096, 1))
    x = Conv1D(16, 17, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(16)(x)
    x = Conv1D(32, 17, activation='relu', padding='same')(x)
    x = MaxPooling1D(16)(x)
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = GlobalAveragePooling1D()(x)  # 替换为全局平均池化
    features = Flatten()(x)
    return Model(inputs=inputs, outputs=features)

# 特征提取网络
def build_feature_extractor_lboro():
    inputs = Input(shape=(4096, 1))
    x = Conv1D(16, 17, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(16)(x)
    x = Conv1D(32, 17, activation='relu', padding='same')(x)
    x = MaxPooling1D(16)(x)
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    # x = MaxPooling1D(2)(x)
    # x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = GlobalAveragePooling1D()(x)  # 替换为全局平均池化
    features = Flatten()(x)
    return Model(inputs=inputs, outputs=features)

# 故障诊断网络
def build_fault_diagnosis_network(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    opt = inputs
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=[opt, outputs])

# 领域判别网络
def build_domain_discriminator(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs=inputs, outputs=outputs)


# 构建 DOC 模型
def build_doc_model(input_shape, num_classes):
    inp = Input(shape=input_shape)
    opt = inp
    features = Dense(128, activation='relu')(inp)
    features = Dropout(0.5)(features)
    outputs = Dense(num_classes, activation='sigmoid')(features)  # One-vs-all sigmoid classifiers
    return Model(inputs=inp, outputs=[opt, outputs])
