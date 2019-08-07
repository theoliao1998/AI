import Networks as Nw
import database_loader as d_l
import time
import copy
import numpy as np



(training_data, validation_data, test_data) = d_l.load_data()
training_data = list(training_data)
validation_data = list(validation_data)
test_data = list(test_data)
names = ["Sigmoid", "tanh", "RELU"]
lmbda = [0.0001,0.03,0.01]
alpha = [0.5,0.005,0.005]
batchsize = [10,25,5]

functs = [[Nw.sigmoid,Nw.sigmoid],[Nw.tanh, Nw.tanh],[Nw.RELU,Nw.RELU]]
de_functs = [[Nw.sigmoid_prime,Nw.sigmoid_prime],[Nw.tanh_prime, Nw.tanh_prime],[Nw.RELU_prime,Nw.RELU_prime]]

N = Nw.Network([784,50,10])


for i in range(3):
    timecost= 0
    n = copy.deepcopy(N)
    n.functs = functs[i]
    n.de_functs = de_functs[i]
    begin = time.time()
    n.training(training_data,100,batchsize[i],alpha[i],lmbda[i],validation_data)
    end = time.time()
    timecost = end - begin
    print(names[i], " function Accuracy :", n.evaluate(test_data)," Time cost :",timecost,"s")

