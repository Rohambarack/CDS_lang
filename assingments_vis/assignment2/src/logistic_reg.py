#important libraries
#setup, importing modules
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Argparse     

# Machine learning 
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
#data
from tensorflow.keras.datasets import cifar10

#functions
def grey_normalize_image(image,bins = 255):

    """ function for greyscaling in an image file, taking it's color histogram and normailizing it. """
    #greyscale
    greyed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #normalize
    hist = cv2.calcHist([greyed_image],[0],None,[bins],[0,256])
    normalized_hist = cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)
    #squeeze a dimension out

    normalized_hist = np.squeeze(normalized_hist)


    return(normalized_hist)

def ch3_normalize_image(image,bins = 255):

    """ function for an image file, taking it's color histograms and normailizing them. """
    color_histogram_list = []
    for channel in range(0,3):

        #normalize
        hist = cv2.calcHist([image],[channel],None,[bins],[0,256])
        normalized_hist = cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)
        #squeeze a dimension out
        normalized_hist = np.squeeze(normalized_hist)
        color_histogram_list.append(normalized_hist)
    
    #make it one array
    color_histogram = np.concatenate(np.array(color_histogram_list))


    return(color_histogram)

def color_profile():
    parser = argparse.ArgumentParser()

    parser.add_argument("--color_profile",
                        "-c_p",
                        required = True,
                        help = "0 for greyscale, 1 to keep colors channels")
    
    parser.add_argument("--bin_size",
                        "-b_s",
                        required = True,
                        help = "How many bins should the color hostograms have. Int value, max 256.")

    args = parser.parse_args()

    return args


def main():
    #data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    #preprocess, grayscale, normalize. bins are set to 64, maybe less is more?
    X_train_normalized_grey =[]
    for image in X_train:
        temp_norm_im = grey_normalize_image(image, bins = 64)
        X_train_normalized_grey.append(temp_norm_im)

    X_test_normalized_grey =[]
    for image in X_test:
        temp_norm_im = grey_normalize_image(image,bins = 64)
        X_test_normalized_grey.append(temp_norm_im)
    
    #classifier
    classifier = LogisticRegression(random_state=42).fit(X_train_normalized_grey, y_train)
    #predict
    prediction = classifier.predict(X_test_normalized_grey)
    #conf_matrix
    cm = np.array2string(metrics.confusion_matrix(y_test,prediction))
    #report
    cr = metrics.classification_report(y_test, prediction)

    #save output
    f = open('../out/log_report.txt', 'w')
    f.write('Logistic Classifier output\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(cr, cm))
    f.close()

if __name__ == "__main__":
    main()