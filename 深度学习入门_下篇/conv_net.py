#coding:utf-8
import cPickle as pickle
import numpy as np
import os
import optim
import matplotlib.pyplot as plt

# 构建一个三层的卷积神经网络，输入层 => （卷积层 => Relu => 池化层） => 全连接 => 输出层(softmax)



def load_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


# 读取数据
def loaddata(root, num_training=5000, num_validation=500, num_test=500):
    # 构建训练集
    xs = []
    ys = []
    for b in range(1, 2):
        f = os.path.join(root, 'data_batch_%d' % (b,))
        X, Y = load_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)

    del X, Y  # 将变量 X,Y 删除

    # 构建测试集
    Xte, Yte = load_batch(os.path.join(root, 'test_batch'))  # 这里是10000条数据

    mask = range(num_training, num_training + num_validation)

    # 将数据划分成 训练集 验证集 测试集
    X_val = Xtr[mask]
    y_val = Ytr[mask]
    mask = range(num_training)
    X_train = Xtr[mask]
    y_train = Ytr[mask]
    mask = range(num_test)
    X_test = Xte[mask]
    y_test = Yte[mask]
    # print X_train.shape

    # 数据预处理，以 0 为中心化
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image  # 这里为什么验证集数据也减去的是 mwan_image
    X_test -= mean_image  # 这里同样的问题

    # 将 通道 的维度放在第二位
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # 返回数据
    return {'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            }
def ReLU(x):
    return np.maximum(0, x)

def affine_forward(x, w, b):
    out = None
    # Reshape x into rows
    N = x.shape[0]
    x_row = x.reshape(N, -1)  # (N,D)
    out = np.dot(x_row, w) + b  # (N,M)
    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache):
    x, w, b = cache
    dx, dw, db = None, None, None
    dx = np.dot(dout, w.T)  # (N,D)
    dx = np.reshape(dx, x.shape)  # (N,d1,...,d_k)
    x_row = x.reshape(x.shape[0], -1)  # (N,D)
    dw = np.dot(x_row.T, dout)  # (D,M)
    db = np.sum(dout, axis=0, keepdims=True)  # (1,M)

    return dx, dw, db

def relu_forward(x):
    out = None
    out = ReLU(x)
    cache = x
    return out, cache

def relu_backward(dout, cache):
    dx, x = None, cache
    dx = dout
    dx[x <= 0] = 0
    return dx

def softmax_loss(x, y):
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

def conv_forward_naive(x, w, b, conv_param):
    stride, pad = conv_param['stride'], conv_param['pad']
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    H_new = 1 + (H + 2 * pad - HH) / stride
    W_new = 1 + (W + 2 * pad - WW) / stride
    s = stride
    out = np.zeros((N, F, H_new, W_new))

    for i in xrange(N):       # ith image
        for f in xrange(F):   # fth filter
            for j in xrange(H_new):
                for k in xrange(W_new):
                    #print x_padded[i, :, j*s:HH+j*s, k*s:WW+k*s].shape
                    #print w[f].shape
                    #print b.shape
                    #print np.sum((x_padded[i, :, j*s:HH+j*s, k*s:WW+k*s] * w[f]))
                    out[i, f, j, k] = np.sum(x_padded[i, :, j*s:HH+j*s, k*s:WW+k*s] * w[f]) + b[f]

    cache = (x, w, b, conv_param)
    return out, cache

def conv_backward_naive(dout, cache):
    #print '1111'
    x, w, b, conv_param = cache
    pad = conv_param['pad']
    stride = conv_param['stride']
    F, C, HH, WW = w.shape
    N, C, H, W = x.shape
    H_new = 1 + (H + 2 * pad - HH) / stride
    W_new = 1 + (W + 2 * pad - WW) / stride

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    s = stride
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    dx_padded = np.pad(dx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    for i in xrange(N):       # ith image
        for f in xrange(F):   # fth filter
            for j in xrange(H_new):
                for k in xrange(W_new):
                    window = x_padded[i, :, j*s:HH+j*s, k*s:WW+k*s]
                    db[f] += dout[i, f, j, k]
                    dw[f] += window * dout[i, f, j, k]
                    dx_padded[i, :, j*s:HH+j*s, k*s:WW+k*s] += w[f] * dout[i, f, j, k]

    # Unpad
    dx = dx_padded[:, :, pad:pad+H, pad:pad+W]

    return dx, dw, db

def max_pool_forward_naive(x, pool_param):
    HH, WW = pool_param['pool_height'], pool_param['pool_width']
    s = pool_param['stride']
    N, C, H, W = x.shape
    H_new = 1 + (H - HH) / s
    W_new = 1 + (W - WW) / s
    out = np.zeros((N, C, H_new, W_new))
    for i in xrange(N):
        for j in xrange(C):
            for k in xrange(H_new):
                for l in xrange(W_new):
                    window = x[i, j, k*s:HH+k*s, l*s:WW+l*s]
                    out[i, j, k, l] = np.max(window)

    cache = (x, pool_param)

    return out, cache


def max_pool_backward_naive(dout, cache):
    x, pool_param = cache
    HH, WW = pool_param['pool_height'], pool_param['pool_width']
    s = pool_param['stride']
    N, C, H, W = x.shape
    H_new = 1 + (H - HH) / s
    W_new = 1 + (W - WW) / s
    dx = np.zeros_like(x)
    for i in xrange(N):
        for j in xrange(C):
            for k in xrange(H_new):
                for l in xrange(W_new):
                    window = x[i, j, k*s:HH+k*s, l*s:WW+l*s]
                    m = np.max(window)
                    dx[i, j, k*s:HH+k*s, l*s:WW+l*s] = (window == m) * dout[i, j, k, l]
    return dx


def affine_relu_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


pass


def conv_relu_forward(x, w, b, conv_param):
    a, conv_cache = conv_forward_naive(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_naive(da, conv_cache)
    return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    a, conv_cache = conv_forward_naive(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_naive(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_naive(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_naive(da, conv_cache)
    return dx, dw, db

# 构建卷积神经网络
class ThreeLayerConvNet(object):
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        # Initialize weights and biases
        C, H, W = input_dim

        # 这里是卷积层参数
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)

        # 这里应该是全连接层
        self.params['W2'] = weight_scale * np.random.randn(num_filters * H * W / 4, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)

        # 输出层 softmax
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)


    # 定义损失函数
    # 损失 = 数值损失 + 权重惩罚项
    def loss(self, X, y=None):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # 前向传播

        # 卷积层参数为：
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # 池化层参数为：
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        a2, cache2 = affine_relu_forward(a1, W2, b2)
        scores, cache3 = affine_forward(a2, W3, b3)


        # 如果这里的 y 值为空，则现在是测试模式
        if y is None:
            return scores

        # 进行反向传播
        data_loss, dscores = softmax_loss(scores, y)
        da2, dW3, db3 = affine_backward(dscores, cache3)
        da1, dW2, db2 = affine_relu_backward(da2, cache2)
        dX, dW1, db1 = conv_relu_pool_backward(da1, cache1)

        # Add regularization
        dW1 += self.reg * W1
        dW2 += self.reg * W2
        dW3 += self.reg * W3
        reg_loss = 0.5 * self.reg * sum(np.sum(W * W) for W in [W1, W2, W3])


        # 计算总 损失
        loss = data_loss + reg_loss

        # 进行权重的更新
        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}

        return loss, grads

class Solver(object):
    def __init__(self, model, data, **kwargs):
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        # Unpack keyword arguments
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 2)
        self.num_epochs = kwargs.pop('num_epochs', 10)

        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in kwargs.keys())
            raise ValueError('Unrecognized arguments %s' % extra)

        # Make sure the update rule exists, then replace the string
        # name with the actual function
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.iteritems()}
            self.optim_configs[p] = d

    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        # Make a minibatch of training data
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # Compute loss and gradient
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        # Perform a parameter update
        for p, w in self.model.params.iteritems():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def check_accuracy(self, X, y, num_samples=None, batch_size=2):
        """
        Check accuracy of the model on the provided data.

        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
          on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using too
          much memory.

        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
          classified by the model.
        """

        # Maybe subsample the data
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Compute predictions in batches
        num_batches = N / batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in xrange(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc

    def train(self):
        """
        Run optimization to train the model.
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train / self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in xrange(num_iterations):
            self._step()

            # Maybe print training loss
            if self.verbose and t % self.print_every == 0:
                print '(Iteration %d / %d) loss: %f' % (
                    t + 1, num_iterations, self.loss_history[-1])

            # At the end of every epoch, increment the epoch counter and decay the
            # learning rate.
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay

            # Check train and val accuracy on the first iteration, the last
            # iteration, and at the end of each epoch.
            first_it = (t == 0)
            last_it = (t == num_iterations + 1)
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(self.X_train, self.y_train,
                                                num_samples=4)
                val_acc = self.check_accuracy(self.X_val, self.y_val, num_samples=4)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)

                if self.verbose:
                    print '(Epoch %d / %d) train acc: %f; val_acc: %f' % (
                        self.epoch, self.num_epochs, train_acc, val_acc)

                # Keep track of the best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.iteritems():
                        self.best_params[k] = v.copy()

        # At the end of training swap the best params into the model
        self.model.params = self.best_params

if __name__ == '__main__':
    data = loaddata('../dataset/cifar-10-batches-py/')

    # 构建一个三层的卷积神经网络
    model = ThreeLayerConvNet(reg=0.9)
    solver = Solver(model, data,
                    lr_decay=0.95,
                    print_every=10, num_epochs=5, batch_size=2,
                    update_rule='sgd_momentum',
                    optim_config={'learning_rate': 5e-4, 'momentum': 0.9})

    solver.train()
    plt.subplot(2, 1, 1)
    plt.title('Training loss')
    plt.plot(solver.loss_history, 'o')
    plt.xlabel('Iteration')

    plt.subplot(2, 1, 2)
    plt.title('Accuracy')
    plt.plot(solver.train_acc_history, '-o', label='train')
    plt.plot(solver.val_acc_history, '-o', label='val')
    plt.plot([0.5] * len(solver.val_acc_history), 'k--')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.gcf().set_size_inches(15, 12)
    plt.show()