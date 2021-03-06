import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt,pi
import math
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys
%matplotlib inline

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    Labels = sorted(np.unique(y))    
    Label_Mean = []
    
    #References:
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html
    #https://engineering.ucsb.edu/~shell/che210d/numpy.pdf
    #find the mean of X for all the  labels..
    #used the formula from here https://web.stanford.edu/class/stats202/content/lec9.pdf 
    covmat = np.cov(X.T)
    for label in Labels:
        lt = []
        index = 0;
        while (index < len(y)):
            if (label == y[index]):
                lt.append(X[index])
            index += 1;
        Label_Mean.append(np.mean(lt, axis = 0))
    means = np.array(Label_Mean).T
    return means, covmat


def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    #References:
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html
    #https://engineering.ucsb.edu/~shell/che210d/numpy.pdf

    # IMPLEMENT THIS METHOD
    Labels = sorted (np.unique(y))
    Mean_Matrix = list()
    Cov_Matrix = []

    # find the the mean matrix and the covriance matrix
    # https://web.stanford.edu/class/stats202/content/lec9.pdf
    for index,label in enumerate(Labels):
        temp_list = []
        index = 0;
        while (index <len(y)):
            if (label == y[index]):
                temp_list.append(X[index])
            index += 1
        Mean_Matrix.append(np.mean(temp_list, axis=0))
        Cov_Matrix.append(np.cov(np.array(temp_list).T))
    
    # Return Mean and Covarience.
    means = np.array(Mean_Matrix).T
    return means, Cov_Matrix

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels
    # https://web.stanford.edu/class/stats202/content/lec9.pdf
    
    y_estimate = []
    label_list = np.array([1,2,3,4,5])
    
    # Use the https://web.stanford.edu/class/stats202/content/lec9.pdf formula
    # 1) Find y estimate
    # 2) find the number of accurate hits
    # 3) Return both y_estimate and the accuracy
    for x in Xtest:
        x = x.reshape(1, 2)
        label_prob = [0] * len(label_list)
        for index,item in  enumerate(label_list):
            u = means[:,index].reshape (1,2)
            mu = u
            sigma = covmat
            Invsigma = inv(sigma)
            power_term =  -1 / 2 * np.matmul((x-u), np.matmul(Invsigma, (x-u).T))
            exp_power_term  = math.exp(power_term)
            label_prob[index] =  ( 1/ ((2*math.pi)**(x.shape[1]/2))* (np.linalg.det(sigma)**(1/2)))*exp_power_term
            #print (label_prob[index])
        y_estimate.append(label_list[label_prob.index(max(label_prob))])
    
    # Calculate accuracy in terms of matches.
    acc_count = 0
    index = 0;
    while (index < len(ytest)):
        if (y_estimate[index] == ytest[index]):
            acc_count += 1
        index += 1
    acc = acc_count / len(y_estimate)
    
    y_estimate = np.array(y_estimate)
    
    return acc, y_estimate

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    y_estimate = []
    label_list = np.array([1,2,3,4,5])

    # Use the https://web.stanford.edu/class/stats202/content/lec9.pdf formula
    # 1) Find y estimate
    # 2) find the number of accurate hits
    # 3) Return both y_estimate and the accuracy

    for x in Xtest:
        x = x.reshape(1, 2)
        label_prob = [0] * len(label_list)
        for index,item in  enumerate(label_list):
            u = means[:,index].reshape (1,2)
            sigma = covmats[index]
            Invsigma = inv(sigma)
            power_term =  -1 / 2 * np.matmul((x-u), np.matmul(Invsigma, (x-u).T))
            exp_power_term  = math.exp(power_term)
            label_prob[index] =  ( 1/ ((2*math.pi)**(x.shape[1]/2))* (np.linalg.det(sigma)**(1/2)))*exp_power_term
            #print (label_prob[index])
        y_estimate.append(label_list[label_prob.index(max(label_prob))])
    #Accuracy
    count = 0
    index = 0;
    while (index < len(y_estimate)):
        if (y_estimate[index] == ytest[index]):
            count += 1
        index += 1
    
    acc = count / len(y_estimate)
    y_esti = np.array(y_estimate)
    return acc, y_esti


def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 

    # Formula to be implemented.
    # w = (xTx)^1 *Xty
    w = np.dot(inv(np.dot(np.transpose(X),X)),np.dot(np.transpose(X),y))
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 

    # Formula to be implemented.
    # w = (xTx)^1 *Xty
    Xt_X = np.dot(np.transpose(X), X)
    LI = lambd*np.eye(Xt_X.shape[0])
    w = np.dot(inv(Xt_X+LI), np.dot(np.transpose(X), y))
    return w
    w = np.dot(inv(np.dot(np.transpose(X),X)),np.dot(np.transpose(X),y))
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    y_est = np.dot(Xtest,w)
    error = y_est - ytest
    #https://mathinsight.org/dot_product_matrix_notation
    mse = np.dot(np.transpose(error),error)/(error.shape[0])
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda   
    
    # implementing the formula
    # 0.5 ((y - Xw)T .(y - Xw) + lambda * wT .W)
    
    d = X.shape[1]
    w_mat = np.reshape(w,(d,1))
    inter = y - np.dot(X_i,w_mat)
    
    error = 0.5*(np.dot(inter.transpose(),inter) + lambd*np.dot(w_mat.transpose(),w_mat))
    
    # diff the same 
    # -0.5 * xT (y-Xw) + lambda * w
    error_grad = -(np.dot(X_i.transpose(),inter)) + lambd*w_mat
    error_grad = np.squeeze(np.array(error_grad))
    
    return error,error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
    # x^0 x^1 X^2 ... x^p
    N  = x.shape[0]
    d  = p + 1
    Xd = np.zeros((N,d))
    # Iterate per colum and take power of x  with the index. yielding 1,x,x^2,x^3... x^p
    for column in range(p+1):
        Xd[:,column] = pow(x,column)
        
    return Xd

## Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))


# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest[:,0])
plt.title('LDA')
plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest[:,0])
plt.title('QDA')


# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest[:,0])
plt.title('LDA')
plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest[:,0])
plt.title('QDA')

# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

print ('Training Data ')
w = learnOLERegression(X,y)
mle = testOLERegression(w,X,y)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,X_i,y)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

print ('Test Data ')
w2 = learnOLERegression(X,y)
mle2 = testOLERegression(w2,Xtest,ytest)

w_i2 = learnOLERegression(X_i,y)
mle_i2 = testOLERegression(w_i2,Xtest_i,ytest)

print('MSE without intercept '+str(mle2))
print('MSE with intercept '+str(mle_i2))


# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
    
index = 0;
print ("lambda  train             test")
while (index < len(mses3)):
    print (np.reshape(lambdas,(lambdas.shape[0],1))[index],mses3_train[index], mses3[index])
    index += 1
    # Break condition we dont  need all  the data
    if (index > 15):
        break;

# hardcoded by looking into the data
opt_lambda = 0.05
opt_w = learnRidgeRegression(X_i,y,opt_lambda)
train_error = testOLERegression(opt_w,X_i,y)
test_error = testOLERegression(opt_w,Xtest_i,ytest)
#print ("\n\n Using optimum lambda 0.05")
#print (" weights : ",opt_w)
#print (" Train Error", train_error," Test Error ",test_error)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.show()



# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1

# find minimum lambda
#index = 0;
#print ("lambda      train mse    test mse")
#while (index < len(w_l)):
    #print (lambdas[index]," ",mses4_train[index], " ", mses4[index])
    #index += 1;
    #if (index > 15):
    #    break;
    
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
#lambda_opt = 0 # REPLACE THIS WITH lambda_opt estimated from Problem 3
lambda_opt = 0.06
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))

plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()

#analysis

index = 0; 
print ("p    Train                 Test                 Train(0.05)              Test(0.05)")
while (index < len(mses5_train[:,0])):
    print (index, "  ",mses5_train[index,0],"   ",mses5[index,0],"  ",mses5_train[index,1],"  ",mses5[index,1])
    index += 1