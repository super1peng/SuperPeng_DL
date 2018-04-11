#coding:utf-8

import cPickle as pickle # 序列化接口
import numpy as np
import os
import optim

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
	Xte, Yte = load_batch(os.path.join(root, 'test_batch')) # 这里是10000条数据

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
	X_val -= mean_image # 这里为什么验证集数据也减去的是 mwan_image
	X_test -= mean_image # 这里同样的问题

	# 将 通道 的维度放在第二位
	X_train = X_train.transpose(0, 3, 1, 2).copy()
	X_val = X_val.transpose(0, 3, 1, 2).copy()
	X_test = X_test.transpose(0, 3, 1, 2).copy()

	# 返回数据
	return {'X_train': X_train, 'y_train':y_train,
			'X_val': X_val, 'y_val': y_val,
			'X_test': X_test, 'y_test': y_test,
			}

def ReLU(x):
    """ReLU non-linearity."""
    return np.maximum(0, x)

def relu_forward(x):
	out = None
	out = ReLU(x)
	cache = x
	return  out, cache
def relu_backward(dout, cache):
	dx, x = None, cache
	dx = dout
	dx[x <= 0] = 0
	return dx

def affine_forward(x, w, b):
	out = None
	N = x.shape[0]
	x_row = x.reshape(N, -1) # 把 x 按照第一个维度展开 N*D
	out = np.dot(x_row, w) + b
	cache = (x, w, b)
	return out, cache

def affine_relu_forward(x, w, b):
	a, fc_cache = affine_forward(x, w, b)
	out, relu_cache = relu_forward(a)
	cache = (fc_cache, relu_cache)
	return out, cache

def softmax_loss(x, y):
	probs = np.exp(x - np.max(x, axis=1, keepdims=True))
	probs /= np.sum(probs, axis=1, keepdims=True)
	N = x.shape[0]
	loss = -np.sum(np.log(probs[np.arange(N), y])) / N
	dx = probs.copy()
	dx[np.arange(N), y] -= 1
	dx /= N
	return loss, dx

def affine_backward(dout, cache):
	x, w, b = cache
	dx, dw, db = None, None, None
	dx = np.dot(dout, w.T)  # (N,D)
	dx = np.reshape(dx, x.shape)  # (N,d1,...,d_k)
	x_row = x.reshape(x.shape[0], -1)  # (N,D)
	dw = np.dot(x_row.T, dout)  # (D,M)
	db = np.sum(dout, axis=0, keepdims=True)  # (1,M)
	return dx, dw, db


def affine_relu_backward(dout, cache):
	fc_cache, relu_cache = cache
	da = relu_backward(dout, relu_cache)
	dx, dw, db = affine_backward(da, fc_cache)
	return dx, dw, db

class TwoLayerNet(object):
	'''
	我们设置隐层的维度为 100
	'''
	def __init__(self, input_dim = 3 * 32 * 32, hidden_dim=100, num_classes =10,
				 weight_scale = 1e-3, reg = 0.0):
		self.params = {}
		self.reg =reg
		self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim) # 第一层的权重参数维度为 3072*100
		self.params['b1'] = np.zeros((1, hidden_dim)) # 第一层的偏置参数的维度为 1 * 100
		self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)  # 第一层的权重参数维度为 100*10
		self.params['b2'] = np.zeros((1, num_classes))  # 第一层的偏置参数的维度为 1 * 10

	def loss(self, X, y=None):  # 损失函数 前向传播
		scores = None
		N = X.shape[0]

		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']

		h1, cache1 = affine_relu_forward(X, W1, b1)  # 第一层添加了Relu
		out, cache2 = affine_forward(h1, W2, b2)     # 第二层并没有添加 Relu

		scores = out  # 这时候神经网络输出的是 num_hidden * num_classes

		# 如果 y 是 None 那么，我们现在处于测试模式
		if y is None:
			return scores

		# 如果不是则计算损失
		loss, grads = 0, {}
		data_loss, dscores = softmax_loss(scores, y) # 这里得到的是数据损失 和 分数
		reg_loss = 0.5 * self.reg * np.sum(W1 * W1) + 0.5 * self.reg * np.sum(W2 * W2)

		loss = data_loss + reg_loss # 得到最终的损失


		# 进行反向传播 计算梯度
		dh1, dW2, db2 = affine_backward(dscores, cache2)
		dX, dW1, db1 = affine_relu_backward(dh1, cache1)

		# 更新参数
		dW2 += self.reg * W2
		dW1 += self.reg * W1
		grads['W1'] = dW1
		grads['b1'] = db1
		grads['W2'] = dW2
		grads['b2'] = db2
		return  loss, grads



class Solver(object):
	def __init__(self, model, data, **kwargs):
		self.model = model
		self.X_train = data['X_train']
		self.y_train = data['y_train']
		self.X_val = data['X_val']
		self.y_val = data['y_val']

		# 解压关键字参数
		self.update_rule = kwargs.pop('update_rule','sgd')
		self.optim_config = kwargs.pop('optim_config', '{}')
		self.lr_decay = kwargs.pop('lr_decay', 1.0)
		self.batch_size = kwargs.pop('batch_size', 100)
		self.num_epochs = kwargs.pop('num_epochs', 10)
		self.print_every = kwargs.pop('print_every', 10)
		self.verbose = kwargs.pop('verbose', True)

		# 如果有超出的参数，则抛出异常
		if len(kwargs) > 0:
			extra = ', '.join('"%s"' % k for k in kwargs.keys())
			raise ValueError('Unrecognized arguments %s' % extra)

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

	def check_accuracy(self, X, y, num_samples=None, batch_size=100):
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
												num_samples=1000)
				val_acc = self.check_accuracy(self.X_val, self.y_val)
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

	model = TwoLayerNet(reg=0.9)
	solver = Solver(model, data,
					lr_decay=0.95,
					print_every=100, num_epochs=50, batch_size=400,
					update_rule='sgd_momentum',
					optim_config={'learning_rate': 5e-4, 'momentum': 0.9})
	solver.train()

	import matplotlib.pyplot as plt
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

	best_model = model
	y_test_pred = np.argmax(best_model.loss(data['X_test']), axis=1)
	y_val_pred = np.argmax(best_model.loss(data['X_val']), axis=1)

	print('Validation set accuracy: ', (y_val_pred == data['y_val']).mean())
	print('Test set accuracy: ', (y_test_pred == data['y_test']).mean())