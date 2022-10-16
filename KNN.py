# coding=utf-8

import operator
import logging
import sys

from numpy import zeros
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format="%(asctime)s %(levelname)s |%(name)s: %(message)s")

logger = logging.getLogger("KNN-Dating")


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
    """
    归一化特征值,消除特征之间量级不同导致的影响 \n
    采用的方法是 https://lqservice.cn/mapping/res/归一化_01.png \n
    :param data_set: 原数据集矩阵 \n
    :return: 归一化后的数据集矩阵
    """
    # numpy 中参数 axis 代表的就是维度 https://blog.csdn.net/weixin_45718167/article/details/104309370
    # 此处我们的 axis 可以取 0 / 1; 经过实际测试 axis 取 0 为按每列取值, axis 取 1 为按每行取值
    min_val_mat = data_set.min(axis=0)
    max_val_mat = data_set.max(axis=0)
    # 最大值矩阵 - 最小值矩阵
    ranges_mat = max_val_mat - min_val_mat
    # 获取矩阵行数
    rows_num = np.shape(data_set)[0]
    # https://www.osgeo.cn/numpy/reference/generated/numpy.tile.html?highlight=tile#numpy.tile
    # tile - 创建一个每一行均为 min_val 的矩阵
    # 求出与最小值的差值矩阵
    norm_data_set_tmp = data_set - np.tile(min_val_mat, (rows_num, 1))
    # 同理 得出 差值 与 最大最小值范围矩阵 的比值矩阵
    norm_data_set_return = norm_data_set_tmp / np.tile(ranges_mat, (rows_num, 1))
    return norm_data_set_return


# 4. KNN 聚类
def classify0(inp_mat, data_set, label_vector, k):
    """
    对于每一个在数据集中的数据点: \n
    1. 计算目标的数据点（需要分类的数据点）与该数据点的距离 \n
    2. 将距离排序: 从小到大 \n
    3. 选取前K个最短距离 \n
    4. 选取这K个中最多的分类类别 \n
    5. 返回该类别来作为目标数据点的预测值 \n
    :return:
    """
    rows_num = np.shape(data_set)[0]
    # 采用欧式距离度量
    # 1. 计算该点与矩阵每一个点的差值
    diff_set = data_set - np.tile(inp_mat, (rows_num, 1))
    # 2. 差值矩阵每个元素进行平方
    square_diff_set = diff_set ** 2
    # 3. 按一维(按行)对差值平方矩阵求和
    sum_vector = square_diff_set.sum(axis=1)
    # 4. 最后不要忘记开根号
    sum_vector = sum_vector ** 0.5
    # 对数组(向量)进行排序 https://www.osgeo.cn/numpy/reference/generated/numpy.argsort.html?highlight=argsort#numpy.argsort
    # 注意此处的返回值代表当前已排序数组的中的元素在原数组中的索引
    # 例如:
    # x = np.array([3, 1, 2])
    # np.argsort(x)
    # 输出 array([1, 2, 0], dtype=int64)
    sorted_sum_vector_index = sum_vector.argsort(kind='quicksort')
    # 创建一个 python 对象作为 Map
    label_map_counter = {}
    for i in range(k):
        # 获取当前数据行在原数据集的标记值
        label = label_vector[sorted_sum_vector_index[i]]
        # 增加计数
        label_map_counter[label] = label_map_counter.get(label, 0) + 1
    # 注意此处 python sorted 高级用法 - (参考 python 对对象进行排序的方法)
    label_map_list = sorted(label_map_counter.iteritems(), key=operator.itemgetter(1), reverse=True)
    return label_map_list[0][0]


if __name__ == '__main__':
    # 指定测试比例
    ho_ratio = 0.1
    # 指定 knn 参数(从多少个最相近的点中判断)
    select_number = 51
    # 读取文件文本数据 -> 矩阵
    data_mat, data_vector = file2matrix('./data/knn_data_set.txt')
    # 分析图表
    # analyse_dating_data(data_mat, data_vector)
    # 归一化原始矩阵
    norm_data_set = auto_norm(data_mat)
    # 总的数据量
    total_data_count = np.shape(norm_data_set)[0]
    # 测试样本数量
    test_data_number = int(total_data_count * ho_ratio)
    # 单一变量可采用 format,多个变量参数建议采用类 c 风格占位符
    logger.info("测试样本数量 {}".format(test_data_number))

    error_count = 0.0

    for i in range(test_data_number):
        to_test_vector_data = norm_data_set[i, :]
        forecast_label = classify0(to_test_vector_data, norm_data_set[test_data_number:total_data_count, :],
                                   data_vector, select_number)
        logger.info("test data [%s]. the forecast label is [%s], the real label is [%s]",
                    to_test_vector_data, forecast_label, data_vector[i])
        if forecast_label != data_vector[i]:
            error_count += 1
    logger.info("total error count [%d]", error_count)
    logger.info("the error rate is [%.2f%%]", (error_count / float(test_data_number)) * 100)

"""
最终测试 随着 select_number 的增加 -> 错误率慢慢下降
"""
