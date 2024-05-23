#import model and libraries
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import re
import pandas as pd
import os
import matplotlib.pyplot as plt

#functions
def filter_dot(path):
    """A function used in loading the image data for assignment3 (and 4), 
    it filters out .git files"""
    #find number of folders in folder
    data_folders = os.listdir(path)
    #remove .gitignore from selection
    regex_data_folders = re.compile(r'^\.')
    filtered = [i for i in data_folders if not regex_data_folders.match(i)]
    
    return filtered

#function for 1 image
def count_faces(image,mtcnn):

    """ Function for taking one image, loading it and extracting the amount of faces in it.
    image : image file path"""

    # Load an image containing faces
    img = Image.open(image)
    # Detect faces in the image
    boxes, _ = mtcnn.detect(img)
    #### get data
    #n faces
    if boxes is None:
        n_faces = 0
    else:
        n_faces = boxes.shape[0] 

    ####extract relevant info from path
    #separate filename from path ( everything after ../data/newspapername/)
    filename = re.findall(r"(?:[^\/]*\/\s*){3}(.*)",image)[0]
    #separate first 3 characters
    newspaper = re.findall(r"^....",filename)[0]
    #separate first number
    year = re.findall(r"^[^\d]*(\d+)",filename)[0]
    #alter the year to decade
    decade = year[:3] + "0s"
    ##### make data into pandas dataframe
    row = pd.DataFrame([[newspaper,decade,year,n_faces]], columns=['NewsPaper','Decade','Year','n_faces'])

    return(row)

#function for 1 folder
def count_faces_folder(folder,mtcnn):

    """Function for iterating the count_faces() function over a folder full of images"""
    files = sorted(os.listdir(folder))

    df_list = [] 

    for picture in files:
        image_path = folder + "/" + picture 

        #image data
        df_row = count_faces(image_path,mtcnn)

        df_list.append(df_row)

    #concat to a single dataframe
    df_full = pd.concat(df_list)

    return(df_full)


#function for all folders
def count_faces_all(mtcnn):

    """Function for iterating count_faces_folder() over all folders. No input is needed, it is setup
    exactly for the file structure of assignment 4 """

    filepath = os.path.join("..","data")
    folders = filter_dot(filepath)

    df_list = []
    for folder in folders:
        current_path = os.path.join(filepath,folder)

        each_newspaper = count_faces_folder(current_path,mtcnn)

        #outputs are also formatted according to the task
        formatted_newspaper = make_df(each_newspaper)

        df_list.append(formatted_newspaper)
    
    overall_df = pd.concat(df_list)

    return overall_df

def make_df(count_output):
   """ A function for taking the raw count output from the model and making it into a 
   presentable dataframe"""

   df = count_output.reset_index()

   #all entries
   all_entries = df.groupby("Decade").count().filter(items =["n_faces"]).rename(columns={"n_faces": "Total"})
   all_values = all_entries["Total"].tolist()
   # 0 face entries
   no_faces = df.loc[df['n_faces'] == 0].groupby("Decade").count().filter(items =["n_faces"]).rename(columns={"n_faces": "Faces"})
   no_face_values = no_faces["Faces"].tolist()
   #group by nespaper and decade to aggreagete rows
   df_fin = df.groupby(["NewsPaper","Decade"]).sum("n_faces").drop(columns = ["index"])
   #add necessary cloumns fro calcualting percentage
   df_fin.insert(1, "Total_Pages", all_values, True)
   df_fin.insert(2, "Page_w_no_Faces", no_face_values, True)
   #calculate percentage and round to 2 decimals
   df_fin["Page_w_Faces"] = df_fin["Total_Pages"] - df_fin["Page_w_no_Faces"]
   df_fin["Percentage"] = df_fin["Page_w_Faces"] / df_fin["Total_Pages"] * 100
   df_fin = df_fin.round({"Percentage" : 2})
   # ungroup data to make visualization easier
   df_fin = df_fin.reset_index()

   return df_fin

#Plotting the data
def comp_n(df):
    #don't alter og
    df_2 = df.copy(deep = True)
    df_3 = df_2[["NewsPaper","Decade","n_faces"]]
    #define index column
    df_3.set_index('Decade', inplace=True)

    df_3.groupby('NewsPaper')['n_faces'].plot(legend=True)
    plt.title("Comparison of Number of Faces in Newspapers")
    output_path = os.path.join("..","out","comp_n.png")
    plt.savefig(output_path)
    plt.close()

#comparison perc
def comp_perc(df):
    #don't alter og
    df_2 = df.copy(deep = True)
    df_3 = df_2[["NewsPaper","Decade","Percentage"]]
    #define index column
    df_3.set_index('Decade', inplace=True)

    df_3.groupby('NewsPaper')['Percentage'].plot(legend=True)
    plt.title("Comparison of Percentages of Pages with Faces in Newspapers")
    output_path = os.path.join("..","out","comp_perc.png")
    plt.savefig(output_path)
    plt.close()


#individual plotting
def ind_n(df,newspaper):
    #subset
    df_n = df[df["NewsPaper"] == newspaper]
    df_n.plot(kind = "line", x = "Decade", y = "n_faces")

    plt.title("Number of faces for newspaper {}".format(newspaper))
    output_path = os.path.join("..","out","n_faces_{}.png".format(newspaper))
    plt.savefig(output_path)
    plt.close()


#individual plotting perc
def ind_perc(df,newspaper):
    #subset
    df_n = df[df["NewsPaper"] == newspaper]
    df_n.plot(kind = "line", x = "Decade", y = "Percentage")

    plt.title("Percentage of Pages with Faces for Newspaper {}".format(newspaper))
    output_path = os.path.join("..","out","percentage_{}.png".format(newspaper))
    plt.savefig(output_path)
    plt.close()





def main():
    #load model
    # Initialize MTCNN for face detection
    mtcnn = MTCNN(keep_all=True)
    # Load pre-trained FaceNet model
    resnet = InceptionResnetV1(pretrained='casia-webface').eval()

    #count faces
    all_faces_df = count_faces_all(mtcnn)
    #save it
    output_path = os.path.join("..","out","Results.csv")
    all_faces_df.to_csv(output_path)
    all_faces_df_2 = pd.read_csv(output_path)
    #comparison
    comp_n(all_faces_df_2)
    comp_perc(all_faces_df_2)
    #individual
    ind_n(all_faces_df_2,"GDL-")
    ind_perc(all_faces_df_2,"GDL-")
    ind_n(all_faces_df_2,"IMP-")
    ind_perc(all_faces_df_2,"IMP-")
    ind_n(all_faces_df_2,"JDG-")
    ind_perc(all_faces_df_2,"JDG-")

    
    

if __name__ == "__main__":
    main()