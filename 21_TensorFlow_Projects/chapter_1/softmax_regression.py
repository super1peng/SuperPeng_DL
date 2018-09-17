#coding:utf-8

# 导入tensorflow
import tensorflow as tf
# 导入数据
from tensorflow.examples.tutorials.mnist import input_data
# 读入数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 创建x占位符(placeholder) 一般这种形式用于存放数据
x = tf.placeholder(tf.float32, [None, 784])  # 张量

# 定义变量 一般用于代表权重 偏置等
W = tf.Variable(tf.zeros([784, 10])) # weights
b = tf.Variable(tf.zeros([10]))

#定义模型的输出
y = tf.nn.softmax(tf.matmul(x,W) + b)

# 真实标签数据,用占位符代替
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

# 有了损失，我们就可以用随机梯度下降针对模型的参数进行优化
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 创建一个session，只有在session中才能才能运行优化步骤
sess = tf.InteractiveSession()

# 对所有变量进行初始化操作，并且分配内存
tf.global_variables_initializer().run()
print('Start training...')

# 进行梯度下降
for _ in range(1000):
    
    # 每一步并不需要训练所有的数据 batch_size
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # 将数据 传入占位符
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 正确的预测结束
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 计算准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 在Session中运行Tensor可以得到Tensor的值
# 这里是获取最终模型的正确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
