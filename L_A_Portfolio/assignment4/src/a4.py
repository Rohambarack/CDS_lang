import os
import pandas as pd
from transformers import pipeline
import numpy as np
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker

def getlabels(line):
    """A function which takes the output from an emotion classifier model,
    collects the labels in it into a list
    
    IMPORTANT: the output of the mode should be a list of dictionaries, with the structure:
    [[{label : "labelname", score : scorevalue},
    {label : "labelname", score : scorevalue}...]]"""

    #the first layer of the output is removed
    abc_unsorted = line[0]
    ### the output is put into alphabetic order
    abc_sorted = sorted(abc_unsorted, key=lambda d: d["label"])
    #the model output is a list of dictionaries, so values are extracted and reformatted
    label_list = []
    #get scores
    for emotion_output in abc_sorted:
        label = emotion_output["label"]
        #add them to their respective lists
        label_list.append(label)
    return label_list

def extract_emotion_scores(line):
    """A function which takes the output from an emotion classifier model, and formats its
    scores to a numpy array for easier handling. 
    line: a single output from an emotional classifier model 

    IMPORTANT: the output of the mode should be a list of dictionaries, with the structure:
    [[{label : "labelname", score : scorevalue},
    {label : "labelname", score : scorevalue}...]]
    """
    #the first layer of the output is removed
    abc_unsorted = line[0]
    ### the output is put into alphabetic order
    abc_sorted = sorted(abc_unsorted, key=lambda d: d["label"])

    #the model output is a list of dictionaries, so values are extracted and reformatted
    score_list = []


    #get scores
    for emotion_output in abc_sorted:
        score = round(emotion_output["score"],2)
        #add them to list
        score_list.append(score)
    #make score list a numpy array, reshape from column to row
    row = np.array(score_list).reshape(1,-1)
    return row

def diag(now,total):
    """quick bug diagnostics
    prints current line being processed, 
    also the total for as a simplified progess bar"""
    
    prog = round(now/total*100,2)

    print("LINE {} at {}%".format(now,prog))

def multiline_extract_emotion_scores(classifier,lines):
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
            prediction_final =  extract_emotion_scores(classifier(line))
        else:
            prediction_temporary =  extract_emotion_scores(classifier(line))

            prediction_final  = np.append(prediction_final,prediction_temporary, axis=0)
        
        #progress + bug diagnostic help
        diag(now = counter+1, total = len(lines))

        counter += 1

        
    return prediction_final

def df_preproc(data):
    """A function for finding the highest predicted emotion from the data and score"""
        
    h_emotion = data[['anger', 'disgust', 'fear', 'joy','neutral', 'sadness', 'surprise']].idxmax(axis=1)
    h_score = data[['anger', 'disgust', 'fear', 'joy','neutral', 'sadness', 'surprise']].max(axis=1)
    data = data.assign(Emotion = h_emotion,)
    data = data.assign(Score = h_score)

    return(data)

def saveplots_season(counted):
    
    fig, axes = plt.subplots(nrows=1, ncols=4)
    counted[counted["Season"] == "Season 1"].plot(ax=axes[0], kind='bar',x="Emotion")
    axes[0].set_title("Season 1")
    counted[counted["Season"] == "Season 2"].plot(ax=axes[1], kind='bar' ,x="Emotion")
    axes[1].set_title("Season 2")
    counted[counted["Season"] == "Season 3"].plot(ax=axes[2], kind='bar' ,x="Emotion")
    axes[2].set_title("Season 3")
    counted[counted["Season"] == "Season 4"].plot(ax=axes[3], kind='bar' ,x="Emotion")
    axes[3].set_title("Season 4")
    fig.tight_layout()
    fig.set_size_inches(7,5)
    plt.savefig(os.path.join("..","out","seasons","S1-4.png"))
    plt.close()

    fig, axes = plt.subplots(nrows=1, ncols=4)
    counted[counted["Season"] == "Season 5"].plot(ax=axes[0], kind='bar',x="Emotion")
    axes[0].set_title("Season 5")
    counted[counted["Season"] == "Season 6"].plot(ax=axes[1], kind='bar' ,x="Emotion")
    axes[1].set_title("Season 6")
    counted[counted["Season"] == "Season 7"].plot(ax=axes[2], kind='bar' ,x="Emotion")
    axes[2].set_title("Season 7")
    counted[counted["Season"] == "Season 8"].plot(ax=axes[3], kind='bar' ,x="Emotion")
    axes[3].set_title("Season 8")
    fig.tight_layout()
    fig.set_size_inches(7,5)
    plt.savefig(os.path.join("..","out","seasons","S5-8.png"))

def saveplots_emotion(freq):

    fig, axes = plt.subplots(nrows=1, ncols=4)
    freq[freq["Emotion"] == "anger"].plot(ax=axes[0], kind='bar',x="Season", y = "Freq")
    axes[0].set_title("anger")
    freq[freq["Emotion"] == "disgust"].plot(ax=axes[1], kind='bar' ,x="Season", y = "Freq")
    axes[1].set_title("disgust")
    freq[freq["Emotion"] == "fear"].plot(ax=axes[2], kind='bar' ,x="Season", y = "Freq")
    axes[2].set_title("fear")
    freq[freq["Emotion"] == "joy"].plot(ax=axes[3], kind='bar' ,x="Season", y = "Freq")
    axes[3].set_title("joy")
    fig.tight_layout()
    fig.set_size_inches(10,7)
    plt.savefig(os.path.join("..","out","emotions","E1.png"))
    plt.close()

    fig, axes = plt.subplots(nrows=1, ncols=3)
    freq[freq["Emotion"] == "neutral"].plot(ax=axes[0], kind='bar',x="Season", y = "Freq")
    axes[0].set_title("neutral")
    freq[freq["Emotion"] == "sadness"].plot(ax=axes[1], kind='bar' ,x="Season", y = "Freq")
    axes[1].set_title("sadness")
    freq[freq["Emotion"] == "surprise"].plot(ax=axes[2], kind='bar' ,x="Season", y = "Freq")
    axes[2].set_title("surprise")
    fig.tight_layout()
    fig.set_size_inches(10,7)
    plt.savefig(os.path.join("..","out","emotions","E2.png"))
    plt.close()

def main():

    #A5###################################
    a5_out = os.path.join("..","..","assignment5","out")
    tracker = EmissionsTracker(project_name="A4",
                            output_dir=a5_out,
                            output_file="emissions_a4.csv")
    ######################################
    #load
    a4_load = tracker.start_task("load")

    #load data
    data_path = os.path.join("..","data")
    data_name = "Game_of_Thrones_Script.csv"
    data_path = os.path.join(data_path,data_name)
    data = pd.read_csv(data_path)

    #load model
    classifier = pipeline("text-classification", 
                      model="j-hartmann/emotion-english-distilroberta-base", 
                      top_k = None)
    
    tracker.stop_task()
    #######################################
    #model
    a4_model = tracker.start_task("model")
    #run model once to get labels
    labels = getlabels(classifier(data["Sentence"][0]))

    #preprocess: remove na values from the sentence row
    #index is reset, so problems are avoided when re-merging dataframes
    data_clean = data.dropna(subset=['Sentence']).reset_index()

    #run model on data to get scores
    predictions = multiline_extract_emotion_scores(classifier,data_clean["Sentence"])

    #make prediction df
    predictions_df = pd.DataFrame(predictions, columns = labels)
    #nicely readd scores to og dataframe
    new_data = data_clean.join(predictions_df)
    #add highest label
    new_data = df_preproc(new_data)

    tracker.stop_task()
    #######################################
    #analys_save
    a4_analyse_save = tracker.start_task("analyse_save")
    #counted occurences
    counted = pd.DataFrame(new_data[["Season","Emotion"]].groupby("Season").value_counts()).reset_index()
    saveplots_season(counted)

    #wrangle by emotions
    wbe = counted.groupby(["Emotion","Season"]).sum().reset_index()
    #get total
    total_df = counted.groupby("Season").sum("count").reset_index()
    #mergeup
    freq = pd.merge(wbe,total_df, on = "Season")
    freq['Freq'] = freq.apply(lambda row: round(row.count_x / row.count_y *100,2), axis=1)
    saveplots_emotion(freq)
    
    
    #######################################
    #save new_data in case further analysis is needed
    output_path = os.path.join("..","out")
    output_df_name = "GoT_emotion_scores.csv"
    full_path = os.path.join(output_path,output_df_name)
    new_data.to_csv(full_path)

    tracker.stop_task()
    _ = tracker.stop()



if __name__ == "__main__":
    main()