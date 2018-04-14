'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    #Code imported from nnScript.py
    return (1.0/ (1.0 + np.exp(-z)))

# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    # Venkat: Set the kth label as 1 . set 0th label 1 for label 0 etc.
    label = np.array(training_label);
    rows = label.shape[0];
    rowsIndex =np.array([i for i in range(rows)])
    training_label = np.zeros((rows,10))
    training_label[rowsIndex,label.astype(int)]=1

    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # Venkat : Add bias and feed forward
    BiasTerm = np.ones(training_data.shape[0])
    training_data = np.column_stack((training_data,BiasTerm))
    num_samples = training_data.shape[0]

    # Venkat: find the output using sigmoid
    HiddenOut = sigmoid(np.dot(training_data,w1.T))

    # Venkat :Add new bias term
    NewBias = np.ones(HiddenOut.shape[0])
    HiddenOutput = np.column_stack((HiddenOut, NewBias))

    # Find the final output using sigmoid
    FinalOutput = sigmoid(np.dot(HiddenOutput,w2.T))


    # Doing back propagation
    delta_l = FinalOutput - training_label

    grad_w2 = np.dot(delta_l.T,HiddenOutput)
    grad_w1 = np.dot(((1-HiddenOutput)*HiddenOutput* (np.dot(delta_l,w2))).T,training_data)


    # remove zero rows hidden
    grad_w1 = np.delete(grad_w1, n_hidden,0)

    # obj_grad
    obj_grad = np.array([])
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = obj_grad/num_samples

    # obj_val
    obj_val_part1 = np.sum(-1*(training_label*np.log(FinalOutput)+(1-training_label)*np.log(1-FinalOutput)))
    obj_val_part1 = obj_val_part1/num_samples
    obj_val_part2 = (lambdaval/(2*num_samples))* ( np.sum(np.square(w1)) + np.sum(np.square(w2)))
    obj_val = obj_val_part1 + obj_val_part2

    return (obj_val,obj_grad)

# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    n=data.shape[0]
    Bias = np.zeros([len(data), 1])
    DataWithBias = np.append(data, Bias ,1)
    HiddenInput = np.dot(DataWithBias ,w1.T)
    HiddenOutput = sigmoid(HiddenInput)

    Bias = np.zeros([len(HiddenOutput), 1])
    FinalDataWithBias = np.append(HiddenOutput, Bias, 1)
    FinalInput = np.dot(FinalDataWithBias, w2.T)
    FinalOutput = sigmoid(FinalInput)
    ans=np.empty((0,1))

    for i in range(n):
        index=np.argmax(FinalOutput[i]);
        ans=np.append(ans,index);
    return ans

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
