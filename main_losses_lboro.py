import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 禁用非错误级别的日志信息
import random as python_random
seed = 42
np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)

# RBF 核函数
def rbf_kernel(X, Y, sigma=1.0):
    X = tf.expand_dims(X, 1)
    Y = tf.expand_dims(Y, 0)
    pairwise_dist = tf.reduce_sum(tf.square(X - Y), axis=-1)
    return tf.exp(-pairwise_dist / (2.0 * sigma ** 2))

# MMD Loss 计算
def mmd_loss(source_features, target_features, sigma=1.0):
    K_ss = rbf_kernel(source_features, source_features, sigma)
    K_tt = rbf_kernel(target_features, target_features, sigma)
    K_st = rbf_kernel(source_features, target_features, sigma)
    return tf.reduce_mean(K_ss) + tf.reduce_mean(K_tt) - 2 * tf.reduce_mean(K_st)

def gaussian_nll(y_true, feat_pred, means, logvars):
    y_true = tf.cast(tf.squeeze(y_true), tf.int32)    # 获取对应类别的均值和 log 方差
    mu_c = tf.gather(means, y_true)  # (batch_size, feature_dim)
    logvar_c = tf.gather(logvars, y_true)  # (batch_size, feature_dim)
    logvar_c = tf.math.softplus(logvar_c)  # 让 logvar_c 始终为正
    # 计算 NLL
    diff = feat_pred - mu_c
    precision = tf.exp(-logvar_c)  # 计算精度 (1 / 方差)
    nll_per_dim = 0.5 * (precision * tf.square(diff) + logvar_c + np.log(2 * np.pi))
    nll = tf.reduce_sum(nll_per_dim, axis=-1)  # 按特征维度求和
    return tf.reduce_mean(nll)  # 求 batch 均值

def separation_loss(means, num_classes, margin):
    total_loss = 0.0
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            dist = tf.norm(means[i] - means[j], ord=2)
            total_loss += tf.maximum(0.0, margin - dist) # + tf.exp(-(dist - margin))
    return total_loss / (num_classes * (num_classes - 1) / 2)

@tf.function
def total_loss(source_labels, source_features, means, logvars, num_classes):
    # 从模型中获取 means 和 logvars
    nll_loss = gaussian_nll(source_labels, source_features, means, logvars)    # 计算负对数似然 (Gaussian NLL)
    # cov_loss = compute_class_covariance_loss(source_features, source_labels, means, num_classes=num_classes)    # 可选：协方差损失或其他正则化项
    combined_loss = nll_loss + separation_loss(means, num_classes, margin=20.0)
    return combined_loss