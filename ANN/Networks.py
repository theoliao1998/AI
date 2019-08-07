import numpy as np
import random
import copy


class Network(object):

    def __init__(self, sizes, functs = None):
        """
        :param: sizes: a list containing the number of neurons in the respective layers of the network.
                See project description.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases2 = copy.deepcopy(self.biases) #if after validation, the accuracy drops, weights will be reloaded back
        self.weights2 = copy.deepcopy(self.weights)
        if functs == None:
            functs = []
            for i in range(len(sizes)-1):
                functs.append(sigmoid)
        else:
            functs = functs[1:]
            
        self.functs = functs
        self.de_functs = []
        for f in functs:  #for more other activation functions, append the corresponding de_functions below
            if f == sigmoid:
                self.de_functs.append(sigmoid_prime)
            elif f == tanh:
                self.de_functs.append(tanh_prime)
            elif f == RELU:
                self.de_functs.append(RELU_prime)
        
    def inference(self, x):
        """
        :param: x: input of ANN
        :return: the output of ANN with input x, a 1-D array
        """
        A = [x]
        for i in range(len(self.biases)):
            z = np.dot(self.weights[i],A[i]) + self.biases[i]
            A.append(self.functs[i](z))
        return A[-1]
        
        

    def training(self, trainData, T, n, alpha, lmbda = None, validationData = None):
        """
        trains the ANN with training dataset using stochastic gradient descent
        :param trainData: a list of tuples (x, y) representing the training inputs and the desired outputs.
        :param T: total number of iteration
        :param n: size of mini-batches
        :param alpha: learning rate
        """
        length = len(trainData)
        
        if validationData != None:
            accuracy = 0
            num = len(validationData)
        
        for i in range(T):
            random.shuffle(trainData)
            miniBatches = [trainData[k:k+n] for k in range(0,length,n)]
            for batch in miniBatches:
                self.updateWeights(batch, alpha, lmbda)
            if validationData != None:
                newaccuracy = self.evaluate(validationData)
                print("iteration: ",i," accuracy ",newaccuracy," / ", num)
                if newaccuracy <= accuracy:
                    self.biases = self.biases2  
                    self.weights = self.weights2
                    break  # early stopping
                else: 
                    self.biases2 = copy.deepcopy(self.biases)
                    self.weights2 = copy.deepcopy(self.weights)
                    accuracy = newaccuracy

    def updateWeights(self, batch, alpha , lmbda = None):
        """
        called by 'training', update the weights and biases of the ANN
        :param batch: mini-batch, a list of pair (x, y)
        :param alpha: learning rate
        """
        n_B = [np.zeros(B.shape) for B in self.biases]
        n_W = [np.zeros(W.shape) for W in self.weights]
        n = len(batch)
        for x,y in batch:
            (nablaW, nablaB) = self.backprop(x,y)
            for i in range(len(n_B)):
                n_B[i] = n_B[i] + nablaB[i]
                n_W[i] = n_W[i] + nablaW[i]
                self.biases[i] = self.biases[i]-(alpha / n) * n_B[i] 
                if (lmbda != None): # L2 regularization, remove the division by the size of the dataset in the gradient update
                    self.weights[i] -= (alpha * lmbda / n) * self.weights[i] 
            
                self.weights[i] = self.weights[i] -(alpha / n) * n_W[i]

    def backprop(self, x, y):
        """
        called by 'updateWeights'
        :param: (x, y): a tuple of batch in 'updateWeights'
        :return: a tuple (nablaW, nablaB) representing the gradient of the empirical risk for an instance x, y
                nablaW and nablaB follow the same structure as self.weights and self.biases
        """
        nablaB = [np.zeros(B.shape) for B in self.biases]
        nablaW = [np.zeros(W.shape) for W in self.weights]
        A = [x]
        Z = []
        for i in range(len(self.biases)):
            z = np.dot(self.weights[i], A[i]) + self.biases[i]
            Z.append(z)
            A.append(self.functs[i](z))
        
        nablaB[-1] = dSquaredLoss(A[-1],y) * self.de_functs[-1](Z[-1])
        nablaW[-1] = np.dot(nablaB[-1], A[-2].transpose())
        for i in range(2,len(self.sizes)):
            nablaB[-i] = np.dot(self.weights[-i+1].transpose(),nablaB[-i+1]) * self.de_functs[-i+1](Z[-i])
            nablaW[-i] = np.dot(nablaB[-i], A[-i-1].transpose())
        return (nablaW,nablaB)
        
            
        

    def evaluate(self, data):
        """
        :param data: dataset, a list of tuples (x, y) representing the training inputs and the desired outputs.
        :return: the number of correct predictions of the current ANN on the input dataset.
                The prediction of the ANN is taken as the argmax of its output
        """
        num = 0
        for (x,y) in data:
            if np.argmax(self.inference(x)) == y:
                num += 1

        return num
        

# activation functions together with their derivative functions:
def dSquaredLoss(a, y):
    """
    :param a: vector of activations output from the network
    :param y: the corresponding correct label
    :return: the vector of partial derivatives of the squared loss with respect to the output activations
    """
    return a - y
    
def sigmoid(z):
    """The sigmoid function"""
    return 1.0 / (1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function"""
    return sigmoid(z)*(1-sigmoid(z))

def tanh(z):  
    return np.tanh(z)  
    
def tanh_prime(z):  
    return 1.0 - np.tanh(z) * np.tanh(z)

def RELU(z):
    return  (z < 0) * 0.01  + (z >= 0) * z
  
def RELU_prime(z):  
    return  (z < 0) * 0.01  + (z >= 0) * 1


"""
nn = Network([3,3,1],[None, sigmoid, sigmoid])
data = [([0,0,1],1),([0,1,2],2),([0,1,1],1),([1,1,2],0),([2,0,1],2),([1,2,0],1),([1,0,1],0),([1,1,1],2),([0,0,0],1)]
nn.training(data,10,3,0.2,None,data[1:-1])
print((nn.evaluate(data))/len(data))
"""
