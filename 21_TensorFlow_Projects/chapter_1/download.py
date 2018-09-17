#coding:utf-8

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 下面手写体数字数据集

from tensorflow.examples.tutorials.mnist import input_data

# 从MNIST_data中读取MNIST数据，这条语句在数据不存在时，会自动执行下载
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 查看训练数据的大小

# 训练集数据
print(mnist.train.images.shape)
print(mnist.train.labels.shape)

# 测试集数据
print(mnist.validation.images.shape)
print(mnist.validation.labels.shape)

# 测试集数据
print(mnist.test.images.shape)
print(mnist.test.labels.shape)

# 查看第0个图片的向量表示
print(mnist.train.images[0,:])

# 查看第0个图片的标签
print(mnist.train.labels[0,:])