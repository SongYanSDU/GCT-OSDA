import numpy as np

def compute_z_scores_for_target(feats_source, label_source, gaussian_model):
    """ 计算目标域的 z-scores 并确定未知样本，同时计算每个类别的 z-score 均值。 """
    num_samples = feats_source.shape[0]
    class_means = gaussian_model.means.numpy()  # (num_classes, feat_dim)
    class_stds = np.exp(gaussian_model.logvars.numpy())  # (num_classes, feat_dim) 确保标准差为正
    num_classes = class_means.shape[0]

    z_scores = np.zeros((num_samples, num_classes))  # 存储所有样本的 z-score

    for c in range(num_classes):
        mean_c = class_means[c]
        std_c = class_stds[c] + 1e-8  # 避免除零错误
        # 计算所有样本对于类别 c 的 z-score
        z_scores[:, c] = np.linalg.norm((feats_source - mean_c) / std_c, axis=1)

    # 计算真实类别的 z-score
    real_z_scores = z_scores[np.arange(num_samples), label_source]
    # 计算每个类别的 z-score 均值
    class_z_score_means = np.zeros(num_classes)

    for c in range(num_classes):
        class_mask = label_source == c  # 找到属于类别 c 的样本
        if np.any(class_mask):  # 仅在类别存在样本时计算均值
            class_z_score_means[c] = np.mean(real_z_scores[class_mask])
    return class_z_score_means

