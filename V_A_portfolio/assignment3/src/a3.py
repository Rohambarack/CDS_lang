#install for ucloud
import os
#packages
import re
# tf tools
import tensorflow as tf
# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
# generic model object
from tensorflow.keras.models import Model
# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD, Adam
#scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
# for plotting
import numpy as np
import matplotlib.pyplot as plt

######### functions
#function in loading 

def filter_dot(path):
    """A function used in loading the image data for assignment3, 
    it filters out .git files"""
    #find number of folders in folder
    data_folders = os.listdir(path)
    #remove .gitignore from selection
    regex_data_folders = re.compile(r'^\.')
    filtered = [i for i in data_folders if not regex_data_folders.match(i)]
    
    return filtered

def preproc_div_255(image):
    "simple preprocessing by dividing pixel color values by max possible value"
    #divides color values in image by max.
    X_new = image.astype("float") / 255

    return X_new

def plot_history(H, epochs):

    """ a function for showing loss curves during training 
    and accuracy in both training and validation sets"""

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.savefig('../out/loss_curve.png')

def load_task_data():
    """specialized image loading for this assignment according to the folder structure"""
    #load image data
    #structure is 1 folder, many folders with pics ordered according to category.
    path1 = os.path.join("..","data")
    folders = filter_dot(path1)
    path2 = os.path.join(path1,folders[0])

    categories = sorted(filter_dot(path2))

        #iterate through all folders with the categories
    list_of_data = []
    for folder in categories:
        current_folder = os.path.join(path2,folder)

        #get files in folder
        files = sorted(filter_dot(current_folder))

        #load image
        for image in files:
            #filter jpegs
            if re.search(r"\.jpg$",image):
            
                current_image_path = os.path.join(current_folder,image)
                current_image = load_img(current_image_path,target_size=(224, 224))
                current_image = img_to_array(current_image)
                #preprocess
                current_image = preproc_div_255(current_image)
                ###########
                label = folder

                data_dict = {"image":current_image,"label":label}

                list_of_data.append(data_dict)
            else:
                pass

    return list_of_data
    
def separate_data(list_of_data):

    """function for separating the data into labels and images"""
    #separate labels and data
    data_list = []
    label_list = []

    for item in list_of_data:
        data = item["image"]
        label = item["label"]

        data_list.append(data)
        label_list.append(label)


    data_list = np.array(data_list)
    
    return data_list, label_list

def label_processing(y_train, y_test):

    """Preprocessing labels for the model"""

    # integers to one-hot vectors
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)

    # initialize label names from folder names
    path1 = os.path.join("..","data")
    folders = filter_dot(path1)
    path2 = os.path.join(path1,folders[0])

    categories = sorted(filter_dot(path2))
    labelNames = categories

    return y_train, y_test, labelNames

def setup_model():
    # load model without classifier layers
    model = VGG16(include_top=False, 
                pooling='avg',
                input_shape=(224, 224, 3))

    #freeze model
    for layer in model.layers:
        layer.trainable = False
    
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    #with batch normalization#
    bn = BatchNormalization()(flat1)
    ##########################
    class1 = Dense(128, activation='relu')(bn)
    output = Dense(10, activation='softmax')(class1)

    # define new model
    model = Model(inputs=model.inputs, 
                outputs=output)
    
    #optimize learning rates
    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.9)
    #set gradient descent
    sgd = SGD(learning_rate=lr_schedule)
    #add to model
    model.compile(optimizer=sgd,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    return model

def main():

    #load data
    data = load_task_data()

    #separate into labels and images
    images, labels = separate_data(data)

    #create splits, stratify on labels
    X_train, X_test, y_train, y_test = train_test_split(images,
                                                    labels, 
                                                    test_size=0.20, 
                                                    random_state=42,
                                                    stratify = labels)

    #binarize labels
    y_train, y_test, labelNames = label_processing(y_train,y_test)

    #setup model
    model = setup_model()

    #data augmentation
    # rotation, scanned images can vary in rotation
    datagen = ImageDataGenerator( 
                                rotation_range=20,
                                validation_split=0.1,
                                )
    
    #fit
    Hist = model.fit(datagen.flow(X_train, y_train, 
                           batch_size=128), 
              validation_data = datagen.flow(X_train, y_train, 
                                             batch_size=128, 
                                             subset = "validation"),
              epochs=25,
              verbose = 1)

    #save model
    model.save("../out/mod_25_epoch.keras")

    #save curves
    plot_history(Hist, 25)

    #save classification report
    predictions = model.predict(X_test, batch_size=128)

    #report
    cr = classification_report(y_test.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=labelNames)
    #save output
    f = open('../out/tobacco_class.txt', 'w')
    f.write('Pretrained image embedding output\n\nClassification Report\n\n{}'.format(cr))
    f.close()

if __name__ == "__main__":
    main()