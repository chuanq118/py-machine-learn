# coding=utf-8
from numpy import zeros
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# 数据集路径 C:\Users\Qi\Documents\机器学习 - 资料\数据集\KNN数据集\datingTestSet2.txt
# 1. 将原数据转为矩阵
def file2matrix(filename):
    """
    导入训练数据 对文本内容 进行一个 map.
    :param filename: 文件路径
    :return: 数据矩阵 returnMat
    """
    f = open(filename)
    lines = f.readlines()
    # zeros 参考 numpy 文档 -> 生成矩阵 / 此处准备返回的矩阵
    return_mat = zeros((len(lines), 3))
    # 准备要返回的标签 张量?
    class_label_vector = []
    index = 0
    for line in lines:
        data_of_one_row = line.strip().split('\t')
        # 对矩阵的每一行进行赋值
        return_mat[index, :] = data_of_one_row[0:3]
        # 添加将每列的类别数据
        class_label_vector.append(int(data_of_one_row[-1]))
        index += 1
    # 返回最终数据
    return return_mat, class_label_vector


# 2. matplotlib 分析数据
def analyse_dating_data(data_mat, data_vector):
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html
    figure = plt.figure()
    # https://matplotlib.org/stable/api/figure_api.html?highlight=add_subplot#matplotlib.figure.Figure.add_subplot
    axes = figure.add_subplot(111)  # 等同于 add_subplot(1, 1, 1)
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html?highlight=scatter#matplotlib.axes.Axes.scatter
    # (x, y, s, c) x,y 为坐标点
    axes.scatter(data_mat[:, 0], data_mat[:, 1], 15 * np.array(data_vector), 15 * np.array(data_vector))
    plt.show()


# 3. 归一化
def auto_norm(data_set):
    # numpy 中参数 axis 代表的就是维度 https://blog.csdn.net/weixin_45718167/article/details/104309370
    # 此处我们的 axis 可以取 0 / 1; 经过实际测试 axis 取 0 为按每列取值, axis 取 1 为按每行取值
    min_val = data_set.min(axis=1)
    max_val = data_set.max(axis=1)
    print min_val, max_val, np.shape(data_set)


if __name__ == '__main__':
    data_mat, data_vector = file2matrix('./data/knn_data_set.txt')
    analyse_dating_data(data_mat, data_vector)
    auto_norm(data_mat)
