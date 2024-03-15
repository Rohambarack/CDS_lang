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
    #preprocess, grayscale, normalize. 
    X_train_normalized_grey =[]
    for image in X_train:
        temp_norm_im = lrg.grey_normalize_image(image)
        X_train_normalized_grey.append(temp_norm_im)

    X_test_normalized_grey =[]
    for image in X_test:
        temp_norm_im = lrg.grey_normalize_image(image)
        X_test_normalized_grey.append(temp_norm_im)
    
    #classifier
    classifier  = MLPClassifier(tol = 0.01,
                           activation = "logistic",
                           hidden_layer_sizes = (100,),
                           max_iter=10,
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

    #loss curve
    plt.plot(classifier.loss_curve_)
    plt.title("Loss curve during training", fontsize=14)
    plt.xlabel('Iterations')
    plt.ylabel('Loss score')
    plt.savefig('../out/loss_curve.png')


if __name__ == "__main__":
    main()