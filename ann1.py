import numpy as np
import random

def tanh_func(z): #tanh actvation function
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

np.random.seed(4)#just to make the same results
input =[0.05, 0.1]
bias1=0.5
bias2=0.7

rand_weight1=np.random.uniform(-0.5,0.5,2)#shape=[w1,w2]
rand_weight2=np.random.uniform(-0.5,0.5,2)#shape=[w3,w4]
rand_weight3=np.random.uniform(-0.5,0.5,2)#shape=[w5,w6]
rand_weight4=np.random.uniform(-0.5,0.5,2)#shape=[w7,w8]
print("Weights for the network\nw1,w2",rand_weight1,"w3,w4",rand_weight2,"w5,w6",rand_weight3,"w7,w8",rand_weight4)

outh1=np.dot(input,rand_weight1)+bias1#hidden layers values
outh2=np.dot(input,rand_weight2)+bias1

h1=tanh_func(outh1)
h2=tanh_func(outh2)

hidden_layer=[h1,h2]
print("Hidden layers  \nh1,h2 ",hidden_layer)
out1=np.dot(hidden_layer,rand_weight3)+bias2#output layers values
out2=np.dot(hidden_layer,rand_weight4)+bias2

out1_tanh=tanh_func(out1)
out2_tanh=tanh_func(out2)
output=[out1_tanh,out2_tanh]
print("Output of the network\nOutput1, Output2",output)
