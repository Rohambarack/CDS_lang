import os
import pandas as pd
from transformers import pipeline
import numpy as np

def getlabels(line):
    """A function which takes the output from an emotion classifier model,
    collects the labels in it into a list"""

    #the first layer of the output is removed
    line = line[0]
    #the model output is a list of dictionaries, so values are extracted and reformatted
    label_list = []
    #get scores
    for emotion_output in line:
        label = emotion_output["label"]
        #add them to their respective lists
        label_list.append(label)
    return label_list

def extract_emotion_scores(line, n_outputs = 7):
    """A function which takes the output from an emotion classifier model, and formats its
    scores to a numpy array for easier handling. 
    line: a single output from an emotional classifier model 

    n_outputs: The function works on the basis, that 7 scores are in the output.

    """
    #the first layer of the output is removed
    line = line[0]
    #the model output is a list of dictionaries, so values are extracted and reformatted
    score_list = []
    #get scores
    for emotion_output in line:
        score = round(emotion_output["score"],2)
        #add them to their respective lists
        label_list.append(label)
        score_list.append(score)
    #make score list a numpy array, reshape from column to row
    row = np.array(score_list).reshape(1,n_outputs)
    return row

def multiline_extract_emotion_scores(classifier,lines, n_outputs = 7):
    """A function for extracting multiple emotion scores from mutiple strings.
    classifier = the classifier used for the emiton score predictions
    lines = the strings which should be predicted by the classifier
    n_outputs = the number of scores that the classifier will return"""

    #counter to help organize arrays, everything is appended to 1st
    counter = 0
    #each line is classified, then scores are extracted as numpy arrays of dimension (1,7)
    #then they are appended so the it becomes (2,7) then (3,7) and so on..
    for line in lines:
        if counter == 0:
            prediction_final =  extract_emotion_scores(classifier(line),n_outputs)
        else:
            prediction_temporary =  extract_emotion_scores(classifier(line),n_outputs)

            prediction_final  = np.append(prediction_final,prediction_temporary, axis=0)
        
        counter += 1
    return prediction_final

def main():

    #A5###################################

    ######################################
    #load

    #load data
    data_path = os.path.join("..","data")
    data_name = "Game_of_Thrones_Script.csv"
    data_path = os.path.join(data_path,data_name)
    data = pd.read_csv(data_path)

    #load model
    classifier = pipeline("text-classification", 
                      model="j-hartmann/emotion-english-distilroberta-base", 
                      return_all_scores=True)
    
    #######################################
    #model

    #run model once to get labels
    labels = getlabels(data["Sentence"][0])

    #run model on data to get scores
    predictions = multiline_extract_emotion_scores(classifier,data["Sentence"])

    #make prediction df
    predictions_df = pd.DataFrame(predictions, columns = labels)
    #nicely readd scores to og dataframe
    new_data = data.join(predictions_df)
    #######################################
    #analyse



    #######################################
    #save

    #save new_data in case further analysis is needed
    output_path = os.path.join("..","out")
    output_df_name = "GoT_emotion_scores.csv"
    full_path = os.path.join(output_path,output_df_name)
    new_data.to_csv(full_path)




if __name__ == "__main__":
    main()