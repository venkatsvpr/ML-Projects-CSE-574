import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC

def preprocess():
    """
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    train_data, labeli = args
    #print ("num samples  and feature -- ",train_data.shape[0],train_data.shape[1])
    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    # Adding bias term
    BiasTerm = np.ones((train_data.shape[0],1))
    InputWithBias = np.column_stack((BiasTerm, train_data))
    # print (" shape ",InputWithBias.shape[0],InputWithBias.shape[1])
    W = initialWeights.reshape(InputWithBias.shape[1],1)
    Theta = np.zeros((InputWithBias.shape[0],1))
    Theta = sigmoid(np.dot(InputWithBias,W))
    # print (" thetaa shape ",Theta.shape[0],Theta.shape[1])
    LogTheta = np.log(Theta)
    y=np.dot(labeli.transpose(),LogTheta)

    # implement the formula for error
    part1 =  np.dot(labeli.transpose(), LogTheta)
    part2 = np.dot(np.subtract(1.0,labeli).transpose(), np.log(np.subtract(1.0,Theta)))
    error = np.sum(part1 + part2)
    error = (-error)/InputWithBias.shape[0]

    # Implement the formula for error_grad
    error_grad = np.zeros((InputWithBias.shape[0], 1))
    part3 = np.zeros((InputWithBias.shape[0],1))
    part3 = np.subtract(Theta,labeli)
    error_grad = np.dot(InputWithBias.transpose(), part3)
    error_grad=error_grad/InputWithBias.shape[0]

    return error, error_grad.flatten()



def blrPredict(W, data):
    Num_of_Samples = data.shape[0]
    label = np.zeros((Num_of_Samples, 1))
    BiasTerm = np.ones((Num_of_Samples,1))
    InputWithBias = np.column_stack((BiasTerm, data))
    # print ("shape is ",InputWithBias.shape[0],InputWithBias.shape[1])
    All_Labels = np.zeros((InputWithBias.shape[0],10))
    All_Labels = sigmoid(np.dot(InputWithBias,W))
    # use argmax and find the lable with the max value
    label = np.argmax(All_Labels, axis =1)
    # print (" shape of label is ",label.shape[0],label.shape[1])
    # reshpae and return the label
    label = label.reshape(Num_of_Samples,1)
    return label

def mlrObjFunction(params, *args):
    train_data, labels = args
    #print (" n ",train_data.shape[0],"features",train_data.shape[0],"labels",labels.shape[1])
    Num_of_Features = train_data.shape[1]
    Num_of_Samples = train_data.shape[0]
    Num_of_Labels = labels.shape[1]

    # Add the bias term
    BiasTerm = np.ones((Num_of_Samples,1))
    InputWithBias = np.column_stack((BiasTerm, train_data))
    # reshape the weights and find exp
    Weights = np.reshape(params,(Num_of_Features + 1, Num_of_Labels))
    exp_WTx = np.exp(np.dot(InputWithBias, Weights))
    weight_sum = np.sum(exp_WTx,axis=1)
    Thetank  = np.transpose(np.divide(np.transpose(exp_WTx), weight_sum))
    # print (" size of thetank ",Thetank.shape[0],Thetank.shape[1])

    # Find the error
    error = 0
    error = np.multiply(labels, np.log(Thetank))
    error = -1 * np.sum(error)/Num_of_Samples
    # Find the error gradient
    error_grad = np.zeros((Num_of_Features + 1, 10))
    error_grad = np.dot(np.transpose(InputWithBias), (Thetank - labels)).flatten()/Num_of_Samples
    return error, error_grad

def mlrPredict(W, data):
    #print (" num samples ",data.shape[0])
    Number_of_Samples = data.shape[0]
    BiasTerm = np.ones((Number_of_Samples,1))
    DataWithBias = np.concatenate((BiasTerm, data), axis = 1)
    # print("datawithbias .. ",DataWithBias.shape[0],DataWithBias.shape[1])
    EstimateLabels = np.zeros((Number_of_Samples,10))
    " Find WT X"
    EstimateLabels = sigmoid(np.dot(DataWithBias, W))
    " Find the max of the all the possiblities"
    label = np.zeros((Number_of_Samples, 1))
    label = np.argmax(EstimateLabels,axis=1)
    " Reshape it to a N*1 "
    label = label.reshape(Number_of_Samples,1)
    # print ("label reshape done ... ",label.shape[0],label.shape[1])
    return label

"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Logistic Regression')
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

"""
Script for Support Vector Machine
"""

print('SVM\nLinear Kernel');
train_label = np.squeeze(train_label)
clf = SVC(kernel = 'linear')
clf.fit(train_data, train_label)
print("Training Accuracy:   "+str(100*clf.score(train_data, train_label))+"%")
print("Test Accuracy:       "+str(100*clf.score(test_data, test_label))+"%")
print("Validation Accuracy: "+str(100*clf.score(validation_data, validation_label))+"%")

print ("\n\nGamma\n");
clf = SVC(kernel='rbf',gamma = 1)
clf.fit(train_data, train_label)
print("Training Accuracy:   "+str(100*clf.score(train_data, train_label))+"%")
print("Test Accuracy:       "+str(100*clf.score(test_data, test_label))+"%")
print("Validation Accuracy: "+str(100*clf.score(validation_data, validation_label))+"%")

print ("\n\nRBF Kernel \n");
clf = SVC(kernel = 'rbf')
clf.fit(train_data, train_label)
print("Training Accuracy:   "+str(100*clf.score(train_data, train_label))+"%")
print("Test Accuracy:       "+str(100*clf.score(test_data, test_label))+"%")
print("Validation Accuracy: "+str(100*clf.score(validation_data, validation_label))+"%")


cvalues  = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
TrainingAccuracy = []
TestAccuracy = []
ValidationAccuracy = []
print ("---------------------------------------------------")
for i in cvalues:
    train_label = np.squeeze(train_label)
    clf = SVC(C=i,kernel='rbf')
    print("Doing for C-Value: ",i)
    clf.fit(train_data, train_label)
    print("CLF Fitting Done!")
    print("Training Accuracy:   ",100*clf.score(train_data, train_label),"%")
    print("Testing Accuracy:    ",100*clf.score(test_data, test_label),"%")
    print("Validation Accuracy: ",100*clf.score(validation_data, validation_label),"%")
    TrainingAccuracy.append(100*clf.score(train_data, train_label))
    TestAccuracy.append(100*clf.score(test_data, test_label))
    ValidationAccuracy.append(100*clf.score(validation_data, validation_label))
    print ("---------------------------------------------------")

accuracyMatrix = np.column_stack((TrainingAccuracy, TestAccuracy, ValidationAccuracy))

#fig = plt.figure(figsize=[12,6])
#plt.subplot(1, 2, 1)
#plt.plot(vector,accuracyMatrix)
#plt.title('Accuracy with varying values of C')
#plt.legend(('Testing data','Training data', 'Validation data'), loc = 'best')
#plt.xlabel('C values')
#plt.ylabel('Accuracy in %')
#plt.show()



"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

print('\n Multiclass logistic Regression');
# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')



