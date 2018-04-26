import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from captcha.image import ImageCaptcha

number = ['0','1','2','3','4','5','6','7','8','9']

def random_captcha_text(char_set=number, captcha_size = 4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text

def gen_captcha_text_and_image():
    image = ImageCaptcha()
    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)

    captcha = image.generate(captcha_text)

    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)

    return captcha_text, captcha_image


def convert2gray(img):
    if len(image.shape) > 2: # 判读如果模型的通道数大于3，则它是一个彩色图需要进行转换
        gray = np.mean(image, -1)
        return gray
    else:
        return img


def text2vec(text): # 将验证码字符串转化为 向量表示 '1234' => [0,1,0,0,0,0,..., 0,0,1,0,0,0.., 0,0,0,1,.., 0,0,0,0,1,0,...]
    text_len = len(text)

    if text_len > MAX_CAPTCHA:
        raise ValueError("验证码最大长度为4")

    vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
    for i,c in enumerate(text):
        idx = i * CHAR_SET_LEN + int(c)
        vector[idx] = 1
    return vector

def vec2text(vec): #将向量转换成验证码文本表示
    text = []
    char_pos = vec.nonzero()[0] # 查找向量中 非零元素的位置
    for i, c in enumerate(char_pos):
        number = i % 10
        text.append(str(number))
    return "".join(text)

# 生成一个训练的batch
def get_next_batch(batch_size=128):  #batch_size 的大小设置为128
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])   # 数据的形式： batch_size * (height * weight)
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])   # 标签的数据形式：batch_size * (验证码长度 * char_set_len)

    # 进行图片的生成操作
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image()
            if image.shape == (60, 160, 3): # 如果图片的大小是 60*160*3则输出
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)

        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y


# 进行cnn网络的定义
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])  #按照tensorflow的数据形式将数据转换


    # 构建一个三层的卷积神经网络 conv1 => relu1 => pooling1 => dropout1 => conv2 => relu2 => pooling2 => dropout2
    # => conv3 => relu3 => pooling3 => dropout3 => fc1 => relu4 => dropout4 => fc2
    # 第二层卷积神经网络
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32])) # 首先对权重和偏置进行初始化操作
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)


    # 第二层卷积神经网络
    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    # 第三层卷积神经网络
    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # 接下来是全连接层
    # 第一层全连接神经网络
    w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    # 第二层全连接神经网络
    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)

    return out

# 对模型进行训练
def train_crack_captcha_cnn():
    output = crack_captcha_cnn() # 引入卷积神经网络结构
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y)) # 定义损失函数
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)  # 优化器设置 adam优化器
    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])  # 预测函数
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            batch_x, batch_y = get_next_batch(64)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            print(step, loss_)

            # 每100 step计算一次准确率
            if step % 10 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print(step, acc)
                # 如果准确率大于50%,保存模型,完成训练
                if acc > 0.50:
                    saver.save(sess, "./model/crack_capcha.model", global_step=step)
                    break

            step += 1


def crack_captcha(captcha_image):
    output = crack_captcha_cnn()  # 导入 cnn结构
    saver = tf.train.Saver() # 导入tensorflow 中的 模型保存读入模型

    with tf.Session() as sess:
        saver.restore(sess, "./model/crack_capcha.model-810") # 将模型各参数导入 设置导入路径

        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={ X:[captcha_image], keep_prob:1 })
        text = text_list[0].tolist()  # 转化成列表类型

        return text

if __name__ == '__main__':
    train = 1
    if train == 0:
        number = ['0','1','2','3','4','5','6','7','8','9']

        text, image = gen_captcha_text_and_image()
        print("验证码图像通道为：",image.shape)

        # 图像参数
        IMAGE_HEIGHT = 60
        IMAGE_WIDTH = 160
        MAX_CAPTCHA = len(text)
        print("验证码文本最长字符数", MAX_CAPTCHA)

    if train == 1:
        number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        IMAGE_HEIGHT = 60
        IMAGE_WIDTH = 160

        char_set = number
        CHAR_SET_LEN = len(char_set)
        # 生成图像
        text, image = gen_captcha_text_and_image()

        # 进行图像显示
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
        plt.imshow(image)
        plt.show()


        MAX_CAPTCHA = len(text)
        image = convert2gray(image) # 将图片转化为灰度图
        image = image.flatten() / 255


        # 设置数据接口
        X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
        Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
        keep_prob = tf.placeholder(tf.float32)  # dropout 设置失活神经元 保留神经元比率

        # 读取训练好的参数
        predict_text = crack_captcha(image)

        print("正确: {}  预测: {}".format(text, predict_text))
