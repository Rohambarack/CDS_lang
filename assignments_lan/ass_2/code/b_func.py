#packages
# system tools
import os
# data  tools
import pandas as pd
import numpy as np

# Machine learning stuff
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

#func

#get data, filter it out, filename, text, label, ts, seed, vectit?
def preprocDF(filename,text,label, fpl = ["..","in"], ts = .2, seed = 123, vectit = 1):
    """ Function for preprocesing and possibly vectorizing data for classifiers.
     filename : name of  .csv file containing text and labels
     text: column with text in the .csv file
     label: column with labels in the .csv file
     fpl: list of folders in relative filepath to file. default is ../in/
     ts: test split percentage. Default is 20%
     seed: seed used for replicative purposes
     vectit: if 1, also vectorizes the data according to :
        - keeps to uni and bigrams,
        - lowercase
        - discards top 5% of most and least common words
        - 500 max features
    
     Two outputs are returned as lists :
        - train_list[text,label]
        - test_data[text,label]
    """
    
    
    #read from in folder
    #loop through all elements in filepath and join ´em
    step_count = 0
    for i in fpl:
        if step_count == 0:
            fp = i
        else:
            fp_temp = os.path.join(fp,i)
            fp = fp_temp

        step_count = step_count + 1


    #final filepath 
    fp_fin = os.path.join(fp,filename)
    #read in df
    df = pd.read_csv(fp_fin)
    #split training and test
    df_train, df_test = train_test_split(df,test_size= ts, random_state= seed)
    x_train, y_train = df_train[text],df_train[label]
    x_test, y_test = df_test[text],df_test[label]

    #vectorise it or not
    if vectit == 0:
        train_list = [x_train,y_train]
        test_list = [x_test,y_test]
    else:
        #create vectorizer specifically for ass_2
        vectorisation = TfidfVectorizer(ngram_range=(1,2),
                        lowercase = True,
                        max_df = .95,
                        min_df = .05,
                        max_features = 500)
        
        #apply vectorizer
        x_train_features = vectorisation.fit_transform(x_train)
        x_test_features = vectorisation.transform(x_test)
        train_list = [x_train_features,y_train]
        test_list = [x_test_features,y_test]
        
    return(train_list,test_list)



    #get data, filter it out, filename, text, label, ts, seed, vectit?
def preprocDF_2(filep,text,label,ts = .2, seed = 123, vectit = 1):
    """ Function for preprocesing and possibly vectorizing data for classifiers.
     filename : name of  .csv file containing text and labels
     text: column with text in the .csv file
     label: column with labels in the .csv file
     fpl: list of folders in relative filepath to file. default is ../in/
     ts: test split percentage. Default is 20%
     seed: seed used for replicative purposes
     vectit: if 1, also vectorizes the data according to :
        - keeps to uni and bigrams,
        - lowercase
        - discards top 5% of most and least common words
        - 500 max features
    
     Two outputs are returned as lists :
        - train_list[text,label]
        - test_data[text,label]
    """
    
    
    #read from in folder
    #final filepath 
    fp_fin = filep
    #read in df
    df = pd.read_csv(fp_fin)
    #split training and test
    df_train, df_test = train_test_split(df,test_size= ts, random_state= seed)
    x_train, y_train = df_train[text],df_train[label]
    x_test, y_test = df_test[text],df_test[label]

    #vectorise it or not
    if vectit == 0:
        train_list = [x_train,y_train]
        test_list = [x_test,y_test]
    else:
        #create vectorizer specifically for ass_2
        vectorisation = TfidfVectorizer(ngram_range=(1,2),
                        lowercase = True,
                        max_df = .95,
                        min_df = .05,
                        max_features = 500)
        
        #apply vectorizer
        x_train_features = vectorisation.fit_transform(x_train)
        x_test_features = vectorisation.transform(x_test)
        train_list = [x_train_features,y_train]
        test_list = [x_test_features,y_test]
        
    return(train_list,test_list)
