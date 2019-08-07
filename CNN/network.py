import numpy as np
from functools import reduce
import math


class Conv2D(object):
    def __init__(self, shape, output_channels, ksize=3, stride=1, method='VALID'):
        self.input_shape = shape
        self.output_channels = output_channels
        self.input_channels = shape[-1]
        self.batchsize = shape[0]
        self.stride = stride
        self.ksize = ksize
        self.method = method

        weights_scale = math.sqrt(reduce(lambda x, y: x * y, shape) / self.output_channels)
        self.weights = np.random.standard_normal(
            (ksize, ksize, self.input_channels, self.output_channels)) / weights_scale
        self.bias = np.random.standard_normal(self.output_channels) / weights_scale

        if method == 'VALID':
            self.eta = np.zeros((shape[0], (shape[1] - ksize + 1) // self.stride, (shape[1] - ksize + 1) // self.stride,
             self.output_channels))

        if method == 'SAME':
            self.eta = np.zeros((shape[0], shape[1]/self.stride, shape[2]/self.stride,self.output_channels))

        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
        self.output_shape = self.eta.shape

        if (shape[1] - ksize) % stride != 0:
            print('input tensor width can\'t fit stride')
        if (shape[2] - ksize) % stride != 0:
            print('input tensor height can\'t fit stride')

    def forward(self, x):
        col_weights = self.weights.reshape([-1, self.output_channels])
        if self.method == 'SAME':
            x = np.pad(x, (
                (0, 0), (self.ksize / 2, self.ksize / 2), (self.ksize / 2, self.ksize / 2), (0, 0)),
                             'constant', constant_values=0)

        self.col_image = []
        conv_out = np.zeros(self.eta.shape)
        for i in range(self.batchsize):
            img_i = x[i][np.newaxis, :]
            self.col_image_i = self.im2col(img_i, self.ksize, self.stride)
            conv_out[i] = np.reshape(np.dot(self.col_image_i, col_weights) + self.bias, self.eta[0].shape)
            self.col_image.append(self.col_image_i)
        self.col_image = np.array(self.col_image)
        return conv_out

    def gradient(self, eta):
        self.eta = eta
        col_eta = np.reshape(eta, [self.batchsize, -1, self.output_channels])

        for i in range(self.batchsize):
            self.w_gradient += np.dot(self.col_image[i].T, col_eta[i]).reshape(self.weights.shape)
        self.b_gradient += np.sum(col_eta, axis=(0, 1))

        # deconv of padded eta with flippd kernel to get next_eta
        if self.method == 'VALID':
            pad_eta = np.pad(self.eta, (
                (0, 0), (self.ksize - 1, self.ksize - 1), (self.ksize - 1, self.ksize - 1), (0, 0)),
                             'constant', constant_values=0)

        if self.method == 'SAME':
            pad_eta = np.pad(self.eta, (
                (0, 0), (self.ksize / 2, self.ksize / 2), (self.ksize / 2, self.ksize / 2), (0, 0)),
                             'constant', constant_values=0)

        flip_weights = np.flipud(np.fliplr(self.weights))
        flip_weights = flip_weights.swapaxes(2, 3)
        col_flip_weights = flip_weights.reshape([-1, self.input_channels])
        col_pad_eta = np.array([self.im2col(pad_eta[i][np.newaxis, :], self.ksize, self.stride) for i in range(self.batchsize)])
        next_eta = np.dot(col_pad_eta, col_flip_weights)
        next_eta = np.reshape(next_eta, self.input_shape)
        return next_eta

    def backward(self, alpha=0.00001, weight_decay=0.0004):
        # weight_decay = L2 regularization
        self.weights *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        self.weights -= alpha * self.w_gradient
        self.bias -= alpha * self.bias

        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)



    def im2col(self,image, ksize, stride):
        # image is a 4d tensor([batchsize, width ,height, channel])
        image_col = []
        for i in range(0, image.shape[1] - ksize + 1, stride):
            for j in range(0, image.shape[2] - ksize + 1, stride):
                col = image[:, i:i + ksize, j:j + ksize, :].reshape([-1])
                image_col.append(col)
        image_col = np.array(image_col)

        return image_col

class BatchNorm(object):
    def __init__(self, shape):
        self.output_shape = shape
        self.batch_size = shape[0]
        self.input_data = np.zeros(shape)

        self.alpha = np.ones(shape[-1])
        self.beta = np.zeros(shape[-1])
        self.a_gradient = np.zeros(shape[-1])
        self.b_gradient = np.zeros(shape[-1])

        self.moving_mean = np.zeros(shape[-1])
        self.moving_var = np.zeros(shape[-1])
        self.epsilon = 0.00001
        self.moving_decay = 0.997

    def forward(self, x, phase='train'):
        self.input_data = x
        self.mean = np.mean(x, axis=(0, 1, 2))
        self.var = self.batch_size / (self.batch_size - 1) * np.var(x,
                                                                    axis=(0, 1, 2)) if self.batch_size > 1 else np.var(
            x, axis=(0, 1, 2))

        # initialize shadow_variable with mean
        if np.sum(self.moving_mean) == 0 and np.sum(self.moving_var) == 0:
            self.moving_mean = self.mean
            self.moving_var = self.var
        # update shadow_variable with mean, var, moving_decay
        else:
            self.moving_mean = self.moving_decay * self.moving_mean  + (1 - self.moving_decay)*self.mean
            self.moving_var = self.moving_decay * self.moving_var + (1 - self.moving_decay)*self.var

        if phase == 'train':
            self.normed_x = (x - self.mean)/np.sqrt(self.var+self.epsilon)
        if phase == 'test':
            self.normed_x = (x - self.moving_mean)/np.sqrt(self.moving_var+self.epsilon)

        return self.normed_x*self.alpha+self.beta

    def gradient(self, eta):
        self.a_gradient = np.sum(eta * self.normed_x, axis=(0, 1, 2))
        self.b_gradient = np.sum(eta * self.normed_x, axis=(0, 1, 2))


        normed_x_gradient = eta * self.alpha
        var_gradient = np.sum(-1.0/2*normed_x_gradient*(self.input_data - self.mean)/(self.var+self.epsilon)**(3.0/2), axis=(0,1,2))
        mean_gradinet = np.sum(-1/np.sqrt(self.var+self.epsilon)*normed_x_gradient, axis=(0,1,2))

        x_gradient = normed_x_gradient*np.sqrt(self.var+self.epsilon)+2*(self.input_data-self.mean)*var_gradient/self.batch_size+mean_gradinet/self.batch_size

        return x_gradient

    def backward(self, alpha=0.0001):
        self.alpha -= alpha * self.a_gradient
        self.beta -= alpha * self.b_gradient


class FullyConnect(object):
    def __init__(self, shape, output_num=2):
        self.input_shape = shape
        self.batchsize = shape[0]

        input_len = reduce(lambda x, y: x * y, shape[1:])

        self.weights = np.random.standard_normal((input_len, output_num))/100
        self.bias = np.random.standard_normal(output_num)/100

        self.output_shape = [self.batchsize, output_num]
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)

    def forward(self, x):
        self.x = x.reshape([self.batchsize, -1])
        output = np.dot(self.x, self.weights)+self.bias
        return output

    def gradient(self, eta):
        for i in range(eta.shape[0]):
            col_x = self.x[i][:, np.newaxis]
            eta_i = eta[i][:, np.newaxis].T
            self.w_gradient += np.dot(col_x, eta_i)
            self.b_gradient += eta_i.reshape(self.bias.shape)

        next_eta = np.dot(eta, self.weights.T)
        next_eta = np.reshape(next_eta, self.input_shape)

        return next_eta

    def backward(self, alpha=0.00001, weight_decay=0.0004):
        # weight_decay = L2 regularization
        self.weights *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        self.weights -= alpha * self.w_gradient
        self.bias -= alpha * self.bias
        # zero gradient
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)

class MaxPooling(object):
    def __init__(self, shape, filtersize=2, stride=2, padding = 'VALID'):
        self.shape_input = shape
        self.filtersize = filtersize
        self.stride = stride
        self.output_shape = list(shape)
        if padding != 'SAME':
            self.output_shape[1] =  (self.shape_input[1] - filtersize) // stride +1
            self.output_shape[2] =  (self.shape_input[2] - filtersize) // stride +1
        else:
            self.output_shape[1] =  math.ceil((self.shape_input[1] - filtersize) / stride) +1
            self.output_shape[2] =  math.ceil((self.shape_input[2] - filtersize) / stride) +1
        self.indices = np.zeros((shape[0],self.output_shape[1],self.output_shape[2],shape[-1],2))
        
        
    def forward(self, x):
        result = np.zeros(self.output_shape)
        for i in range(x.shape[0]):
            for j in range(self.shape_input[-1]):
                r_prime = 0
                for r in range(0,self.output_shape[1]*self.stride, self.stride):
                    c_prime = 0
                    for c in range(0,self.output_shape[2]*self.stride, self.stride):
                        result[i,r_prime,c_prime,j] = np.max(x[i,r:(r+self.filtersize), c:(c+self.filtersize),j])
                        index = np.argmax(x[i,r:(r+self.filtersize), c:(c+self.filtersize),j])
                        self.indices[i,r_prime,c_prime,j] = np.array([r + index//self.stride, c + index%self.stride])
                        c_prime += 1
                    r_prime += 1
        return result
    
    def gradient(self, deltas):
        result = np.zeros(self.shape_input)
        for i in range(deltas.shape[0]):
            for j in range(self.shape_input[-1]):
                for y_prime in range(deltas.shape[1]):
                    for x_prime in range(deltas.shape[2]):
                        (y,x) = self.indices[i,y_prime,x_prime,j]
                        y = int(y)
                        x = int(x)
                        result[i,y,x,j] = deltas[i,y_prime,x_prime,j]
            
        return result

class Relu(object):
    def __init__(self, shape):
        self.eta = np.zeros(shape)
        self.x = np.zeros(shape)
        self.output_shape = shape

    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def gradient(self, eta):
        self.eta = eta
        self.eta[self.x<0]=0
        return self.eta


class Softmax(object):
    def __init__(self, shape):
        self.softmax = np.zeros(shape)
        self.eta = np.zeros(shape)
        self.batchsize = shape[0]

    def cal_loss(self, prediction, label):
        self.label = label
        self.prediction = prediction
        self.predict(prediction)
        self.loss = 0
        for i in range(self.batchsize):
            self.loss += np.log(np.sum(np.exp(prediction[i]))) - prediction[i, label[i]]

        return self.loss

    def predict(self, prediction):
        exp_prediction = np.zeros(prediction.shape)
        self.softmax = np.zeros(prediction.shape)
        for i in range(self.batchsize):
            prediction[i, :] -= np.max(prediction[i, :])
            exp_prediction[i] = np.exp(prediction[i])
            self.softmax[i] = exp_prediction[i]/np.sum(exp_prediction[i])
        return self.softmax

    def gradient(self):
        self.eta = self.softmax.copy()
        for i in range(self.batchsize):
            self.eta[i, self.label[i]] -= 1
        return self.eta
