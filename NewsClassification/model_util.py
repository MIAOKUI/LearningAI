from sklearn.model_selection import train_test_split
import os
import jieba
import glob
import numpy as np
import sklearn.feature_extraction.text as t2v
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import time



def get_filelist(data_dir):
    """
    :param data_dir: raw data directory
    :return: A list of news file
    """

    file_list = glob.glob(os.path.join(data_dir, '*_*.txt'))
    return file_list

def words_parsing(file_list):
    """
    :param file_list: A list or news files
    :return:
        class_label: text label list of each news
        words: news record after parsing
    """
    class_label = []
    words = []
    for f in file_list:
        with open(f, 'rb', ) as fh:
            each_news = ''.join([l.decode(encoding='utf-8').strip() for l in fh.readlines()])
        class_label.append(os.path.basename(f).split('_')[0].lower())
        parsed_words = ' '.join(list(jieba.cut(each_news)))
        words.append(parsed_words)
    return class_label, words

def to_vector_Tfidf(parsed_words):
    """

    :param parsed_words: string list of parsed by jieba
    :return: vectors transformed using Tf-IDF algorithms
    """
    vectorizer = t2v.TfidfVectorizer(max_features=10000)
    vectors = vectorizer.fit_transform(parsed_words)
    return vectors


def to_vector_countVectors(parsed_words):
    vectorizer = t2v.CountVectorizer(max_df=0.95, min_df=2, max_features=10000)
    vectors = vectorizer.fit_transform(parsed_words)
    return vectors


def data_prepare(data_dir, vectorizer):
    """
    :param data_dir: raw data director
    :return: vectors labels and d_label(label dict)
    """
    file_list = get_filelist(data_dir)
    class_label, words = words_parsing(file_list)
    vectors = vectorizer(words)
    s_label = set(class_label)
    d_label = dict(zip(s_label, range(len(s_label))))
    labels = [d_label[k] for k in class_label]
    return vectors, labels, d_label


def lr_model(train_data, train_label, test_data, test_label, penalty, C):
    lr = LogisticRegression(multi_class='multinomial', solver='newton-cg', penalty=penalty, C=C)
    train_pred, test_pred, train_score, test_score =_performance(lr, train_data, train_label, test_data, test_label)
    return train_pred, test_pred, train_score, test_score


def svm(train_data, train_label, test_data, test_label):
    sv = SVC(kernel='rbf')
    train_pred, test_pred, train_score, test_score =_performance(sv, train_data, train_label, test_data, test_label)
    return train_pred, test_pred, train_score, test_score


def bayes_model(train_data, train_label, test_data, test_label):
    gnb = GaussianNB()
    train_pred, test_pred, _, _ =_performance(gnb, train_data.toarray(), train_label, test_data.toarray(), test_label)
    return train_pred, test_pred


def rd_model(train_data, train_label, test_data, test_label, n, depth):
    rd = RandomForestClassifier(n_estimators=n, max_depth=depth)
    train_pred, test_pred, _, _ =_performance(rd, train_data, train_label, test_data, test_label)
    return train_pred, test_pred


def gbdt_model(train_data, train_label, test_data, test_label,n):
    gbdt = GradientBoostingClassifier()
    train_pred, test_pred, _, _ =_performance(gbdt, train_data, train_label, test_data, test_label)
    return train_pred, test_pred


def ada_model(train_data, train_label, test_data, test_label, n):
    ada = AdaBoostClassifier(n_estimators=n)
    train_pred, test_pred, _, _ =_performance(ada, train_data, train_label, test_data, test_label)
    return train_pred, test_pred


def nn_model(train_data, train_label, test_data, test_label, hidden_layer_sizes = 4, alpha=0.0001):
    nn = MLPClassifier(hidden_layer_sizes, activation='relu', alpha=alpha)
    train_pred, test_pred, _, _ =_performance(nn, train_data, train_label, test_data, test_label)
    return train_pred, test_pred


def LDA(train_data, test_data, n_topics):
    lda = LatentDirichletAllocation(n_topics, n_jobs=4)
    lda.fit(train_data)
    train_data_new = lda.transform(train_data)
    test_data_new = lda.transform(test_data)
    return train_data_new, test_data_new


def _performance(md, train_data, train_label, test_data, test_label):
    fit_begin = time.time()
    md.fit(train_data, train_label)
    fit_time = time.time() - fit_begin
    train_score = md.score(train_data, train_label)
    test_score = md.score(test_data, test_label)
    train_pred_begin = time.time()
    train_pred = md.predict(train_data)
    train_pred_time = time.time() - train_pred_begin
    test_pred_begin = time.time()
    test_pred = md.predict(test_data)
    test_pred_time = time.time() - test_pred_begin
    print("Predict accuracy: train "+str(train_score) + ", test: " + str(test_score))
    print("Running time: fit " + str(fit_time) + ", Predict on Train: " + str(train_pred_time) + ", Predict on Test: " + str(test_pred_time))
    return train_pred, test_pred, train_score, test_score



if __name__== '__main__':
    ## 使用Tfidf进行向量化
    data1, labels1, label_dict1 = data_prepare("data/news", to_vector_Tfidf)
    train_data1, test_data1, train_label1, test_label1 = train_test_split(data1, labels1)

    ## 使用tf进行向量化
    data2, labels2, label_dict2 = data_prepare("data/news", to_vector_countVectors)
    train_data2, test_data2, train_label2, test_label2 = train_test_split(data1, labels1)

    ## 测试两种方式机基模型性能
    #1. tifidf
    _, _, _, _=lr_model(train_data1, train_label1, test_data1, test_label1, penalty='l2', C=1)

    #2. tf
    _, _, _, _ = lr_model(train_data2, train_label2, test_data1, test_label1, penalty='l2', C=1)




