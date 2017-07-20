"""
from src import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
from src import network
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
"""

"""
from src import network2
net = network2.Network([784,30,10],cost = network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data,30,10,0.5,evaluation_data=test_data,monitor_evaluation_accuracy=True)
"""
import csv
import numpy as np
window = 120
epochs = 30
mini_batch_size = 50
eta = 5



with open('.\data\AAPL.csv') as csvfile:
    trainData = csv.reader(csvfile, delimiter=',')
    priceList = []
    for row in trainData:
        priceList.append(float(row[2]))
priceList = np.asarray(priceList)

returnList = priceList[window:-1] / priceList[(window-1):-2] - 1

return_index = np.ones(len(returnList))
return_index[returnList >= 0.02] = 0
return_index[returnList <= -0.02] = 2

train_data = []*len(return_index)
for i in range(len(return_index)):
    index = np.zeros((3,1))
    index[int(return_index[i])] = 1
    train_data.append([np.reshape(priceList[i:(i+window)],(window,1)),index])


with open('.\data\AAPL_test.csv') as csvfile:
    testData = csv.reader(csvfile, delimiter=',')
    priceList = []
    for row in testData:
        priceList.append(float(row[2]))
priceList = np.asarray(priceList)

returnList = priceList[window:-1] / priceList[(window-1):-2] - 1

return_index = np.ones(len(returnList))
return_index[returnList >= 0.02] = 0
return_index[returnList <= -0.02] = 2

test_data = []*len(return_index)
for i in range(len(return_index)):
    test_data.append([np.reshape(priceList[i:(i+window)],(window,1)),return_index[i]])

from src import network
net = network.Network([window,20,10, 3])
net.SGD(train_data, epochs, mini_batch_size, eta, test_data=test_data)
