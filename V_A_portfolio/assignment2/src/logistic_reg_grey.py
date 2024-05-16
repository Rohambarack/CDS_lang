#important libraries
#setup, importing modules
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd    

# Machine learning 
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
#data
from tensorflow.keras.datasets import cifar10

#functions
def grey_normalize_image(image):

    """ function for greyscaling an image file, and normailizing it. """
    #greyscale
    greyed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #normalize
    normalized = greyed_image/255.0
    #reshape
    normalized = normalized.reshape(-1, 1024)
    #squeeeze out a dimension
    normalized = np.squeeze(normalized)


    return(normalized)

def main():
    #data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    #preprocess, grayscale, normalize.
    X_train_normalized_grey =[]
    for image in X_train:
        temp_norm_im = grey_normalize_image(image)
        X_train_normalized_grey.append(temp_norm_im)

    X_test_normalized_grey =[]
    for image in X_test:
        temp_norm_im = grey_normalize_image(image)
        X_test_normalized_grey.append(temp_norm_im)
    
    classifier = LogisticRegression( 
                         solver='saga',
                         multi_class='multinomial',
                         random_state = 42).fit(X_train_normalized_grey, y_train)
    #predict
    prediction = classifier.predict(X_test_normalized_grey)
    
    #labels
    labels = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

    #report
    cr = metrics.classification_report(y_test, prediction, target_names = labels)

    #save output
    f = open('../out/log_grey_report.txt', 'w')
    f.write('Logistic Classifier output\n\nClassification Report\n\n{}'.format(cr))
    f.close()

if __name__ == "__main__":
    main()