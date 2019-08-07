import numpy as np
import copy
import math
from functools import reduce

class Conv(object):
    def __init__(self, shape, filter_num, filtersize=3, stride=1, padding='VALID'):
        """filter size should be odd"""
        self.shape_input = shape #(batchsize, length, height, channelnum)
        self.filter_num = filter_num
        self.filtersize = filtersize
        self.stride = stride
        self.method = padding
        self.shape_output = list(shape)
        self.shape_output[-1] = filter_num # output channel num = filter num
        if padding != 'SAME':
            self.shape_output[1] =  (self.shape_output[1] - filtersize) // stride +1
            self.shape_output[2] =  (self.shape_output[2] - filtersize) // stride +1
        else:
            self.shape_output[1] =  math.ceil((self.shape_input[1] - filtersize) / stride) +1
            self.shape_output[2] =  math.ceil((self.shape_input[2] - filtersize) / stride) +1
        
        weights_scale = math.sqrt((filtersize**2)*shape[-1]/2)
        self.weights = np.random.standard_normal((filtersize,filtersize,shape[-1],filter_num)) / weights_scale
        self.biases = np.random.standard_normal(filter_num) / weights_scale
        

    def forward(self, x):
        self.x = x
        self.col_image = []
        col_weights = self.weights.reshape([-1, self.shape_output[-1]])
        if self.method == "SAME":
            double_padding = ((self.shape_output[1]-1) * self.stride - (self.shape_input[1] - self.filtersize)) 
            if double_padding % 2 == 0:
                padding1 = (double_padding//2, double_padding//2)
            else:
                padding1 = ((double_padding-1)//2, (double_padding+1)//2)
            double_padding = ((self.shape_output[2]-1) * self.stride - (self.shape_input[2] - self.filtersize)) 
            if double_padding % 2 == 0:
                padding2 = (double_padding//2, double_padding//2)
            else:
                padding2 = ((double_padding-1)//2, (double_padding+1)//2)
      
            x = np.pad(x,((0, 0), padding1, padding2, (0, 0)),'constant')
            self.padding = (padding1, padding2)    
        

        result = np.zeros(tuple(self.shape_output))
        for i in range(self.shape_output[0]):
            img_i = x[i][np.newaxis, :]
            self.col_image_i = im2col(img_i, self.filtersize, self.stride)
            result[i] = np.reshape(np.dot(self.col_image_i, col_weights) + self.biases, self.shape_output[1:])
            self.col_image.append(self.col_image_i)
        self.col_image = np.array(self.col_image)

        return result
        
    
    def backward(self, deltas, alpha, lmbda = 0):
        pad_deltas = copy.deepcopy(deltas)
        deltas_prime = np.reshape(deltas, [self.shape_output[0], -1, self.shape_output[-1]])
        
        if(self.stride > 1):
            i = 1
            while(i < pad_deltas.shape[1]):
                for j in range(self.stride-1):
                    v = np.zeros((1,pad_deltas.shape[2],pad_deltas.shape[3]))
                    pad_deltas = np.insert(pad_deltas,i,v,axis=1)
                i += self.stride
            i = 1
            #print(deltas[0,:,:,0])
            while(i < pad_deltas.shape[2]):
                for j in range(self.stride-1):
                    v = np.zeros((1,pad_deltas.shape[3]))
                    pad_deltas = np.insert(pad_deltas,i,v,axis=2)
                i += self.stride
        
        pad_deltas = np.pad(pad_deltas, ((0, 0), (self.filtersize-1, self.filtersize-1), (self.filtersize-1, self.filtersize-1), (0, 0)),'constant')
        
        """
        weights_flipped = np.zeros(self.weights.shape)
        for i in range(self.weights.shape[-1]):
            for j in range(self.weights.shape[-2]):
                weights_flipped[:,:,j,i] = self.weights[:,:,j,i][::-1]
                for k in range(self.weights.shape[0]):
                    weights_flipped[:,:,j,i][k] = weights_flipped[:,:,j,i][k][::-1]
        
  
        shape = list(pad_deltas.shape)
        shape[1] = shape[1]-self.filtersize + 1
        shape[2] = shape[2]-self.filtersize + 1
        shape[-1] = self.shape_input[-1]
        result = np.zeros(shape)

        for i in range(self.shape_output[0]):
            self.conv(pad_deltas[i], result[i], weights_flipped, 1, 'BACKWARD')
            #print("res",result[i])
        #print(result[0,:,:,0])
        if self.method != 'SAME':
            z = np.zeros(self.shape_input)
            z[:,0:result.shape[1],0:result.shape[2],:] = result
            result = z
        else:
            (padding1,padding2) = self.padding
            (a,b) = padding1
            b = result.shape[1]-b
            (c,d) = padding2
            d = result.shape[2]-d
            result = result[:,a:b,c:d,:]
        """
        flip_weights = np.flipud(np.fliplr(self.weights))
        flip_weights = flip_weights.swapaxes(2, 3)
        col_flip_weights = flip_weights.reshape([-1, self.shape_input[-1]])
        col_pad_deltas = np.array([im2col(pad_deltas[i][np.newaxis, :], self.filtersize, self.stride) for i in range(self.shape_input[0])])
        result = np.dot(col_pad_deltas, col_flip_weights)
        result = np.reshape(result, self.shape_input)


        ## update ##
        deltas_w = np.zeros(self.weights.shape)
        deltas_b = np.zeros(self.biases.shape)
        col_deltas = np.reshape(deltas, [self.shape_input[0], -1, self.shape_output[-1]])

        for i in range(self.shape_input[0]):
            deltas_w += np.dot(self.col_image[i].T, col_deltas[i]).reshape(self.weights.shape)
        deltas_b += np.sum(col_deltas, axis=(0, 1))
        
        self.weights *= 1 - lmbda
        self.weights -= alpha * deltas_w
        #print(self.weights[:,:,0,0])
        self.biases -= lmbda * deltas_b

        return result
        
def im2col(image, filtersize, stride):
    # image is a 4d tensor([batchsize, width ,height, channel])
    image_col = []
    for i in range(0, image.shape[1] - filtersize + 1, stride):
        for j in range(0, image.shape[2] - filtersize + 1, stride):
            col = image[:, i:i + filtersize, j:j + filtersize, :].reshape([-1])
            image_col.append(col)
    image_col = np.array(image_col)

    return image_col 
 

class MaxPooling(object):
    def __init__(self, shape, filtersize=2, stride=2, padding = 'VALID'):
        self.shape_input = shape
        self.filtersize = filtersize
        self.stride = stride
        self.shape_output = list(shape)
        if padding != 'SAME':
            self.shape_output[1] =  (self.shape_input[1] - filtersize) // stride +1
            self.shape_output[2] =  (self.shape_input[2] - filtersize) // stride +1
        else:
            self.shape_output[1] =  math.ceil((self.shape_input[1] - filtersize) / stride) +1
            self.shape_output[2] =  math.ceil((self.shape_input[2] - filtersize) / stride) +1
        self.indices = np.zeros((shape[0],self.shape_output[1],self.shape_output[2],shape[-1],2))
        
        
    def forward(self, x):
        result = np.zeros(self.shape_output)
        for i in range(x.shape[0]):
            for j in range(self.shape_input[-1]):
                r_prime = 0
                for r in range(0,self.shape_output[1]*self.stride, self.stride):
                    c_prime = 0
                    for c in range(0,self.shape_output[2]*self.stride, self.stride):
                        result[i,r_prime,c_prime,j] = np.max(x[i,r:(r+self.filtersize), c:(c+self.filtersize),j])
                        index = np.argmax(x[i,r:(r+self.filtersize), c:(c+self.filtersize),j])
                        self.indices[i,r_prime,c_prime,j] = np.array([r + index//self.stride, c + index%self.stride])
                        c_prime += 1
                    r_prime += 1
        return result
    
    def backward(self, deltas):
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
        self.deltas = np.zeros(shape)
        self.x = np.zeros(shape)
        self.shape_output = shape

    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, deltas):
        self.detas = deltas
        self.deltas[self.x<0]=0
        return self.deltas


class FC(object):
    def __init__(self, shape, output_num = 2):
        self.shape_input = shape
        self.biases = np.random.standard_normal(output_num)/100
        input_num = reduce(lambda x, y: x * y, shape[1:])
        self.weights = np.random.standard_normal((input_num, output_num))/100
        #print(self.weights)
        self.shape_output = [shape[0], output_num]
    
    def forward(self, x):
        self.x = x.reshape([self.shape_input[0], -1])
        result = np.dot(self.x, self.weights)+self.biases
        return result
 
    def backward(self, deltas, alpha, lmbda = 0.0001):
		## update ##
        deltas_w = np.zeros(self.weights.shape)
        deltas_b = np.zeros(self.biases.shape)
        n = self.shape_output[0]
        
        for i in range(n):
            deltas_w += np.dot(self.x[i][:, np.newaxis], deltas[i][:, np.newaxis].T)
            deltas_b += deltas[i].reshape(self.biases.shape)
        
        self.weights *= (1 - lmbda)
        self.weights -= alpha * deltas_w
        #print(self.weights[:,:,0,0])
        self.biases -= lmbda * deltas_b

        result = np.dot(deltas, self.weights.T)
        result = np.reshape(result, self.shape_input)
        
        return result

class Softmax(object):
    def __init__(self, shape):
        self.shape_input = shape #[batchsize, num]

    def cross_entropy(self, prediction, validation):
        self.validation = validation
        self.prediction = prediction
        self.predict(prediction)
        self.loss = 0
        for i in range(self.shape_input[0]):
            self.loss += np.log(np.sum(np.exp(prediction[i]))) - prediction[i, validation[i]]

        return self.loss

    def predict(self, x):
        self.result = np.zeros(x.shape)
        x_exp = np.zeros(x.shape)
        for i in range(self.shape_input[0]):
            x[i, :] -= np.max(x[i, :]) # avoid overflow
            #x_exp[i] = np.exp(x[i])
            self.result[i] = np.exp(x[i])/np.sum(np.exp(x[i]))
        return self.result
    
    def backward(self):
        self.deltas = self.result.copy()
        for i in range(self.shape_input[0]):
            self.deltas[i, self.validation[i]] -= 1
        return self.deltas



