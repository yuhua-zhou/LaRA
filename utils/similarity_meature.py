import numpy as np


def cosine_similarity(X, Y):
    # 计算矩阵 X 和 Y 的 L2 范数
    norm_X = np.linalg.norm(X, axis=1, keepdims=True)
    norm_Y = np.linalg.norm(Y, axis=1, keepdims=True)

    # 计算矩阵内积
    dot_product = X.dot(Y.T)

    # 计算余弦相似度
    cosine_sim = dot_product / (norm_X * norm_Y.T)

    # 返回平均余弦相似度
    return np.mean(cosine_sim)


def cka_similarity(X, Y):
    def centered_gram(K):
        n = K.shape[0]
        unit = np.ones([n, n])
        return K - unit.dot(K) / n - K.dot(unit) / n + unit.dot(K).dot(unit) / n ** 2

    def frobenius_norm(K):
        return np.sqrt(np.sum(K ** 2))

    Kx = X.T.dot(X)
    Ky = Y.T.dot(Y)

    gKx = centered_gram(Kx)
    gKy = centered_gram(Ky)

    cka = frobenius_norm(gKx.T.dot(gKy)) / (frobenius_norm(gKx) * frobenius_norm(gKy))

    return cka


def svd_similarity(X, Y):
    _, s_x, _ = np.linalg.svd(X)
    _, s_y, _ = np.linalg.svd(Y)

    # 计算向量的 L2 范数
    norm_v1 = np.linalg.norm(s_x)
    norm_v2 = np.linalg.norm(s_y)

    # 计算向量内积
    dot_product = np.dot(s_x, s_y)

    # 计算余弦相似度
    cosine_sim = dot_product / (norm_v1 * norm_v2)

    return cosine_sim


# 假设有两个特征矩阵 X 和 Y
X = np.random.randn(256, 256)
Y = np.random.randn(256, 256)

# X = np.array([
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9]
# ])
#
# Y = np.array([
#     [10, 2, 3],
#     [4, 5, 6],
#     [7, 8, 10]
# ])

# 计算 CKA 相似度
similarity = cka_similarity(X, Y)
print(f"CKA similarity: {similarity:.4f}")

similarity = cosine_similarity(X, Y)
print(f"Cosine similarity: {similarity:.4f}")

similarity = svd_similarity(X, Y)
print(f"SVD similarity: {similarity:.4f}")
