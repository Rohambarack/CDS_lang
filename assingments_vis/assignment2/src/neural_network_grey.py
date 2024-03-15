#setup, importing modules
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#functions
import logistic_reg_grey as lrg

# Machine learning 
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
#data
from tensorflow.keras.datasets import cifar10

def main():
    #data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    #preprocess, grayscale, normalize. bins are set to 64, maybe less is more?
    X_train_normalized_grey =[]
    for image in X_train:
        temp_norm_im = lrg.grey_normalize_image(image, bins = 64)
        X_train_normalized_grey.append(temp_norm_im)

    X_test_normalized_grey =[]
    for image in X_test:
        temp_norm_im = lrg.grey_normalize_image(image,bins = 64)
        X_test_normalized_grey.append(temp_norm_im)
    
    #classifier
    classifier  = MLPClassifier(activation = "logistic",
                           hidden_layer_sizes = (20,),
                           max_iter=1000,
                           random_state = 42)
    
    
    classifier.fit(X_train_normalized_grey, y_train)


    #predict
    prediction = classifier.predict(X_test_normalized_grey)
    #conf_matrix
    cm = np.array2string(metrics.confusion_matrix(y_test,prediction))
    #report
    cr = metrics.classification_report(y_test, prediction)

    #save output
    f = open('../out/NN_grey_report.txt', 'w')
    f.write('Neural Network Classifier output\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(cr, cm))
    f.close()

if __name__ == "__main__":
    main()