from Conv import *
from base_conv import Conv2D
from FC import FullyConnect
import numpy as np

import time
import struct
from glob import glob


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    images_path = glob('./%s/%s*3-ubyte' % (path, kind))[0]
    labels_path = glob('./%s/%s*1-ubyte' % (path, kind))[0]

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


images, labels = load_mnist('./data/mnist')
test_images, test_labels = load_mnist('./data/mnist', 't10k')
print(images.shape, labels.shape)

#images = images[:1000,:]
#labels = labels[:1000]
print(images.shape, labels.shape)


batch_size = 100
conv1 = Conv2D([batch_size, 28, 28, 1], 12, 5, 1)
relu1 = Relu(conv1.output_shape)
pool1 = MaxPooling(relu1.shape_output)
conv2 = Conv2D(pool1.shape_output, 24, 3, 1)
relu2 = Relu(conv2.output_shape)
pool2 = MaxPooling(relu2.shape_output)
fc = FullyConnect(pool2.shape_output, 10)
sf = Softmax(fc.output_shape)


learning_rate = 0.0001
lmbda = 0.0004 

# train_loss_record = []
# train_acc_record = []
# val_loss_record = []
# val_acc_record = []
acc = 0.7
batch_acc = 0
for epoch in range(20):
    # if epoch < 5:
    #     learning_rate = 0.00001
    # elif epoch < 10:
    #     learning_rate = 0.000001
    # else:
    #     learning_rate = 0.0000001
    

            
    batch_loss = 0
    batch_acc = 0
    val_acc = 0
    val_loss = 0

    # train
    train_acc = 0
    train_loss = 0
    print(images.shape)
    acc = 0.7
    for i in range(int(images.shape[0] / batch_size)):
        """
        if (i-1)%10 == 0:
            print(acc," ", batch_acc / float( batch_size))
        if (batch_acc / float(batch_size) - acc) >= 0.1:
            learning_rate /= 10
            lmbda /= 10
            acc += 0.1
        elif acc >= 0.8:
            if(batch_acc / float(batch_size) - acc) >= 0.05:
                learning_rate /= 10
                lmbda /= 10
                acc += 0.05    
        elif acc >= 0.9:
            if(batch_acc / float(batch_size) - acc >= 0.02):
                learning_rate /= 10
                lmbda /= 10
                acc += 0.02
        """         
        batch_loss = 0
        batch_acc = 0
        
        img = images[i * batch_size:(i + 1) * batch_size].reshape([batch_size, 36, 36, 3])
        label = labels[i * batch_size:(i + 1) * batch_size]
        conv1_out = relu1.forward(conv1.forward(img))
        pool1_out = pool1.forward(conv1_out)
        conv2_out = relu2.forward(conv2.forward(pool1_out))
        pool2_out = pool2.forward(conv2_out)
        fc_out = fc.forward(pool2_out)
        batch_loss += sf.cross_entropy(fc_out, np.array(label))
        train_loss += sf.cross_entropy(fc_out, np.array(label))

        for j in range(batch_size):
            if np.argmax(sf.result[j]) == label[j]:
                batch_acc += 1
                train_acc += 1

        if i % 1 == 0:
            deltas = sf.backward()
            deltas = fc.backward(deltas, learning_rate, lmbda)
            deltas = pool2.backward(deltas)
            deltas = relu2.backward(deltas)
            deltas = conv2.backward(deltas,learning_rate, lmbda)
            deltas = pool1.backward(deltas)
            deltas = relu1.backward(deltas)
            deltas = conv1.backward(deltas,learning_rate, lmbda)

            if i % 10 == 0: 
                print("epoch ",epoch," batch ",i," acc ", batch_acc / float(batch_size)," loss ", batch_loss / batch_size, learning_rate)
                

            

    print("epoch ",epoch," batch ",i," acc ", train_acc / float(images.shape[0])," loss ", train_loss / images.shape[0])


"""
    # validation
    for i in range(int(test_images.shape[0] / batch_size)):
        img = test_images[i * batch_size:(i + 1) * batch_size].reshape([batch_size, 28, 28, 1])
        label = test_labels[i * batch_size:(i + 1) * batch_size]
        conv1_out = relu1.forward(conv1.forward(img))
        pool1_out = pool1.forward(conv1_out)
        conv2_out = relu2.forward(conv2.forward(pool1_out))
        pool2_out = pool2.forward(conv2_out)
        fc_out = fc.forward(pool2_out)
        val_loss += sf.cross_entropy(fc_out, np.array(label))

        for j in range(batch_size):
            if np.argmax(sf.result[j]) == label[j]:
                val_acc += 1
    print("epoch ",epoch," acc ", val_acc / float(test_images.shape[0])," loss ", val_loss / test_images.shape[0])
    
    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "  epoch: %5d , val_acc: %.4f  avg_val_loss: %.4f" % (
        epoch, val_acc / float(test_images.shape[0]), val_loss / test_images.shape[0])
"""
