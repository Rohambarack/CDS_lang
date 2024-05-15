#setup, importing modules
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#adding option (using VGG16 instrad of image color histograms)
from numpy.linalg import norm
from tqdm import notebook
# tensorflow
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import (load_img, 
                                                  img_to_array)
from tensorflow.keras.applications.vgg16 import (VGG16, 
                                                 preprocess_input)
from sklearn.neighbors import NearestNeighbors

import argparse
### argparse arguments
def input_parser():
    parser =  argparse.ArgumentParser(description='Spotify assignment inputs')
    parser.add_argument('--image',
                        "-i",
                        default = "image_0121.jpg",
                        help='image file name (optional)')
    parser.add_argument('--VGG16',
                        "-v",
                        default = 0,
                        help='method. default is based on color histograms')
    
    args = parser.parse_args()
    return args


#functions
def normit(image):

    """ function for reading in an image file, taking it's color histograms and normailizing them. """
    hist = cv2.calcHist([image],[0,1,2],None,[255,255,255],[0,256,0,256,0,256])
    normalized_hist = cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)
    return(normalized_hist)

def count_hist_dist(filename,name_list,image_list,n_returns):

    """function for finding Distance from a set image file's color histograms.
    filename: the name of the file, which will be the target
    name_list: a list of all filenames the comparison should include
    image_list: a list of images. the images are loaded in, with three colour channels and two spacial
    dimensions. The list of images should have the same length and order as the list of names.
    n_returns: (int) how many filenames and distances should be returned."""

    #find target according file index
    target_ind = name_list.index(filename) 
    #create target comparison hist
    target = normit(image_list[target_ind])

    #compare to target
    comp_list =[]
    for i in image_list:
        norm =  normit(i)
        histcomp = round(cv2.compareHist(target,norm,cv2.HISTCMP_CHISQR), 2)
        #save comparison values
        comp_list.append(histcomp)

    #assign filenames to comparison values

    df_pd = pd.DataFrame({"Filename":name_list})
    df_pd["Distance"] = comp_list

    #sort and filter
    #sort ascending
    df_sorted = df_pd.sort_values(axis=0, by= "Distance", ascending= True)
    #drops 1st entry (it's the target image themselves)
    df_sorted = df_sorted.query('Distance != 0')
    #keeps 5 closest images
    df_fin = df_sorted[:n_returns]

    return(df_fin)

#new processing function for pretrained model
def extract_features(img_path, model):
    """
    Extract features from image data using pretrained model (e.g. VGG16)
    """
    # Define input image shape - remember we need to reshape
    input_shape = (224, 224, 3)
    # load image from file path
    img = load_img(img_path, target_size=(input_shape[0], 
                                          input_shape[1]))
    # convert to array
    img_array = img_to_array(img)
    # expand to fit dimensions
    expanded_img_array = np.expand_dims(img_array, axis=0)
    # preprocess image 
    preprocessed_img = preprocess_input(expanded_img_array)
    # use the predict function to create feature representation
    features = model.predict(preprocessed_img, verbose=False)
    # flatten
    flattened_features = features.flatten()
    # normalise features
    normalized_features = flattened_features / norm(features)
    return normalized_features

#save feature extracted results
def vgg16_results(n,target_image,n_feature_list,name_list):
    """ Function for saving the results from the pre trained model for the top 5 
    most similar pictures
    n: amount of similar pictures
    target_image: name of target image file
    n_feature_list: list of image representations
    name_list:list of all image names"""

    #target indice of image_0121.jpg
    targ = name_list.index(target_image)
    #find its neighbours
    neighbors = NearestNeighbors(n_neighbors=10, 
                                algorithm='brute',
                                metric='cosine').fit(n_feature_list)
    #find distance and index
    distances, indices = neighbors.kneighbors([n_feature_list[targ]])
    #find n neighbours, exclude picture itself
    idxs = []
    dists = []
    for i in range(1,n+1):
        dists.append(distances[0][i])
        idxs.append(indices[0][i])
    #find picture names from indeces
    VGG16_names = []
    for index in idxs:
        image_name = name_list[index]

        VGG16_names.append(image_name)

    df_pd = pd.DataFrame({"Filename":VGG16_names})
    df_pd["Distance"] = dists

    #sort and filter
    #sort ascending
    df_sorted = df_pd.sort_values(axis=0, by= "Distance", ascending= True)
    
    return(df_sorted)

#function for saving image version:
def save_visual_results(target_name,df,method):

    """ Saving visual representaions for easier comparison of results
    target_name: target image file
    df: dataframe with image file names
    method: which mathod vas used, color histograms or vgg, used in nameing output file"""
    #make pics
    target_pic_path =os.path.join("..","in",target_name)
    target_pic = cv2.imread(target_pic_path)

    #list of 5 closest
    close_list = []
    for pic in df["Filename"]:

        fp = os.path.join("..","in",pic)
        image = cv2.imread(fp)

        close_list.append(image)

    #create multiplot
    f, axarr = plt.subplots(3,2)
    axarr[0,0].imshow(target_pic)
    axarr[0, 0].set_title("TARGET")
    axarr[0,1].imshow(close_list[0])
    axarr[1,0].imshow(close_list[1])
    axarr[1,1].imshow(close_list[2])
    axarr[2,0].imshow(close_list[3])
    axarr[2,1].imshow(close_list[4])

    # Hide x labels and tick labels 
    for ax in axarr.flat:
        ax.tick_params(left = False, right = False , labelleft = False , 
                    labelbottom = False, bottom = False) 
    
    #save figure
    figure_path = os.path.join("..","out",target_name + "_" + method + ".png")
    f.savefig(figure_path)
        

##### main function
def main():
    #argparse args
    args = input_parser()
    
    #load in files
    #loop load all flowers and names and save it
    fp = os.path.join("..","in")
    fp_s = sorted(os.listdir(fp))


    image_list = []
    name_list = []
    for i in fp_s:
        filepath = os.path.join(fp,i)
        image = cv2.imread(filepath)
        names = i
        name_list.append(i)
        image_list.append(image)

    #define target image and method
    target_name = args.image
    method_vgg = args.VGG16

    if method_vgg == 0:
        method = "color_hist"
        #Find distances for e.g. target file "image_0121.jpg"
        
        target_path = os.path.join("..","in",target_name)
        count_df = count_hist_dist(target_name,name_list,image_list,5)


        #save output
        out_p = os.path.join("..","out",target_name + "_comp.csv")
        count_df.to_csv(out_p, index=False)

        #save visual output

        save_visual_results(target_name,count_df,method)
    else:

        method = "vgg16"
        model = VGG16(weights='imagenet', 
              include_top=False,
              pooling='avg',
              input_shape=(224, 224, 3))


        root_dir = os.path.join("..","in")
        filenames = [root_dir + "/" + name for name in sorted(os.listdir(root_dir))]
        n_feature_list = []
        # iterate over all files with a progress bar
        for i in notebook.tqdm(range(len(filenames))):
            n_f = extract_features(filenames[i], model)
            n_feature_list.append(n_f)

        df_vgg = vgg16_results(5,target_name,n_feature_list,name_list)
        
        #save output
        out_p = os.path.join("..","out",target_name + "_comp_vgg.csv")
        df_vgg.to_csv(out_p, index=False)

        #save visual output

        save_visual_results(target_name,df_vgg,method)

if __name__ == "__main__":
    main()
