import numpy as np

data_files = np.load("NewsClassification/data/words.npz")
datas = data_files['data']
labels = data_files['label']
print(labels)
# 建立字典，标签向量化
s_labels = set(labels)
d_labels = dict(zip(s_labels, range(len(s_labels))))
labels = np.array([d_labels[itr] for itr in labels])
# print(s_labels)
# print(d_labels)
print(labels)
# 文本向量化
# vect_tool = TfidfVectorizer(max_features=10000)
# vect = vect_tool.fit_transform(datas)
# # LDA降维
# lda = LatentDirichletAllocation(n_components=30, learning_method='batch')
# compress_data = lda.fit_transform(vect)
# # 分类，逻辑回归是及格线
# method = LogisticRegression()
# method.fit(compress_data, labels)
# pd = method.predict(compress_data)
# # 预测准确度
# print(np.sum(pd==labels)/len(labels))