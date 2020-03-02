# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:53:05 2020

@author: Frank
"""

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import copy, random

# Sigmoid function
def logistic(x):
    ## Done: 1)
    return 1/(1+np.exp(-x))

def hypothesisLogistic(X, coefficients, bias):
    ## TODO: 2)
    lambda_t = np.zeros(X.shape[1])
    h_x = np.zeros(X.shape[0])
    for instance in range(0,X.shape[0]):
        for feat in range(0,X.shape[1]):
            lambda_t[feat] = X[instance][feat]*coefficients[feat]
        h_x[instance] = lambda_t.sum() + bias
    return logistic(h_x)



def gradient_descent_log(bias, coefficients, alpha, X, Y, max_iter):

    length = len(Y)
    lossFunc = np.zeros(X.shape[0])
    # array is used to store change in cost function for each iteration of GD
    errorValues = []
    
    
    for num in range(0, max_iter):
        
        # Calculate predicted y values for current coefficient and bias values 
        predictedY = hypothesisLogistic(X, coefficients, bias)
        

        # calculate gradient for bias
        biasGrad =    (1.0/length) *  (np.sum( predictedY - Y))
        
        #update bias using GD update rule
        bias = bias - (alpha*biasGrad)
        
        # for loop to update each coefficient value in turn
        for coefNum in range(len(coefficients)):
            
            # calculate the gradient of the coefficient
            gradCoef = (1.0/length)* (np.sum( (predictedY - Y)*X[:, coefNum]))
            
            # update coefficient using GD update rule
            coefficients[coefNum] = coefficients[coefNum] - (alpha*gradCoef)
        
        #TODO: 3)
        # Cross Entropy Error 
        cost = 0
        #Cross Entropy Error for a single Instance
        for instance in range(0,predictedY.shape[0]):
            if Y[instance] == 1:
                lossFunc[instance] = -(Y[instance]*np.log(predictedY[instance])) 
            elif (Y[instance] == 0) & (predictedY[instance] != 1):
                lossFunc[instance] = -(1-Y[instance])*np.log(1-predictedY[instance])
            elif (Y[instance] == 0) & (predictedY[instance] == 1):
                lossFunc[instance] = 20

        #Average Cross entropy error for all predicted values
        cost =  (1/Y.shape[0])*lossFunc.sum()

        errorValues.append(cost)

    
    # plot the cost for each iteration of gradient descent
    plt.plot(errorValues)
    plt.show()
    
    return bias, coefficients, errorValues, predictedY

def calculateAccuracy(bias, coefficients, X_test, y_test):
    #calculateAccuracy
    cost = 0
    lossFunc = np.zeros(X_test.shape[0])
    predictedY = hypothesisLogistic(X_test, coefficients, bias)
    
    
    
    for z in range(0,predictedY.shape[0]):
        if predictedY[z] > 0.5:
            predictedY[z] = 1
        elif predictedY[z] <= 0.5:
            predictedY[z] = 0
    
    #MSE
    MSE = np.zeros(predictedY.shape[0])
    MSE = (predictedY-y_test)**2
    sumMSE = (1/(predictedY.shape[0]+2))*(MSE.sum())
    print ("Final MSE value:",+ sumMSE)
    correct = np.zeros(predictedY.shape[0])
    correct = predictedY == y_test
    #Accuracy
    print("Percentage of correctly predicted values : ",  str((correct.sum()/predictedY.shape[0])*100),"%")
    return MSE, str((correct.sum()/predictedY.shape[0])*100)

def logisticRegression(X_train, y_train, X_test, y_test,max_iter,alpha):

    # set the number of coefficients equal to the number of features
    #and randomly initialize values
    coefficients = np.zeros(X_train.shape[1])
    for z in range(0,X_train.shape[1]):
        coefficients[z] = random.uniform(0,2)
    bias = random.uniform(0,5)
   
    # call gredient decent, and get intercept(bias) and coefficents
    bias, coefficients, errorValues, predictedYresults = gradient_descent_log(bias, coefficients, alpha, X_train, y_train, max_iter)
    MSE, Accuracy = calculateAccuracy(bias, coefficients, X_test, y_test)
    
    return MSE, Accuracy
    
#%%
def main():
    
    digits = datasets.load_digits()
    
    # Display one of the images to the screen
    plt.figure(1, figsize=(3, 3))
    plt.imshow(digits.images[3], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()
    
    # Load the feature data and the class labels
    X_digits = digits.data
    y_digits = digits.target
    
    # The logistic regression model will differentiate between two digits
    # Code allows you specify the two digits and extract the images 
    # related to these digits from the dataset
    indexD1 = y_digits==1
    indexD2 = y_digits==7
    allindices = indexD1 | indexD2
    X_digits = X_digits[allindices]
    y_digits = y_digits[allindices]
 

    # We need to make sure that we conveert the labels to 
    # 0 and 1 otherwise our cross entropy won't work 
    lb = preprocessing.LabelBinarizer()
    y_digits = lb.fit_transform(y_digits)
    y_digits  =y_digits.flatten()

    n_samples = len(X_digits)

    
    # Seperate data in training and test
    # Training data 
    X_train = X_digits[:int(.7 * n_samples)]
    y_train = y_digits[:int(.7 * n_samples)]
    
    # Test data
    X_test = X_digits[int(.7 * n_samples):]
    y_test = y_digits[int(.7 * n_samples):]

   
    max_iter = 100
    alpha = 0.1
    MSE, Accuracy = logisticRegression(X_train,y_train,X_test,y_test,max_iter,alpha)
     
main()
