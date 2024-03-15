#setup, importing modules
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#functions
import logistic_reg_non_grey as lrg

# Machine learning 
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
#data
from tensorflow.keras.datasets import cifar10

def main():
    #data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
   #preprocess, normalize, pu color channel histoggrams after oneanother. bins are set to 64, maybe less is more?
    X_train_ch3 =[]
    for image in X_train:
        temp_norm_im = lrg.ch3_normalize_image(image, bins = 64)
        X_train_ch3.append(temp_norm_im)

    X_test_ch3 =[]
    for image in X_test:
        temp_norm_im = lrg.ch3_normalize_image(image,bins = 64)
        X_test_ch3.append(temp_norm_im)
    
    #classifier
    classifier  = MLPClassifier(activation = "logistic",
                           hidden_layer_sizes = (50,),
                           max_iter=1000,
                           random_state = 42)
    
    
    classifier.fit(X_train_ch3, y_train)


    #predict
    prediction = classifier.predict(X_test_ch3)
    #conf_matrix
    cm = np.array2string(metrics.confusion_matrix(y_test,prediction))
    #report
    cr = metrics.classification_report(y_test, prediction)

    #save output
    f = open('../out/NN_non_grey_report.txt', 'w')
    f.write('Neural Network Classifier output\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(cr, cm))
    f.close()

if __name__ == "__main__":
    main()