
import os
os.system("pip install pandas spacy")
import spacy
os.system("python -m spacy download en_core_web_md")
nlp = spacy.load('en_core_web_md')
import pandas as pd
import numpy as np
import re


# function for 1 file from 1 folder
def count_ofof(fp, extract_this = ["NOUN","VERB","ADJ","ADV"], ent_list = ["PERSON","GPE","ORG"], clean_metadata = '<.*?>'):

    """A function for basic text mining for 1 text file from 1 folder.
    fp = filepath of the exact text file,
    extract_this = list of PartOfSpeech values in spacy.pos_ syntax,
        defaults are ["NOUN","VERB","ADJ","ADV"]
    ent_list = list of Named entity values in spacy.ent.label_ syntax,
        defaults are ["PERSON","GPE","ORG"]
    clean_metadata = regex specification to remove metadata
        defaults are '<.*?>' 
    Relies on additional functions:
     - addmissing(extract,list_of_stuff)"""

    #load it with encoding 
    with open(fp, encoding="latin-1") as f:
        text = f.read()

    #clean metadata 
    text_c = re.sub(clean_metadata, '', text)

    #convert it with spacy
    text_nlp = nlp(text_c)

    #extract values, I have to use "i" to  index items, long words confuse me
    #empty list for extracts
    extract = []
    for i in text_nlp:
        if i.pos_ in extract_this:
            extract.append(i.pos_)
        else:
            pass 
    
    #get named entities out
    ent_extract = []
    for i in text_nlp.ents:
        if i.label_ in ent_list:
            ent_extract.append(i.label_)
        else:
            pass
    
    #add lists if there were none
    missing_pos, missing_pos_zeros = addmissing(extract,extract_this)
    missing_ents, missing_ents_zeros = addmissing(ent_extract, ent_list)

    #regroup
    extract = extract + ent_extract
    missing_pos = missing_pos + missing_ents
    missing_pos_zeros = missing_pos_zeros + missing_ents_zeros

    #returning counts of unique occurences
    if len(missing_pos) == 0:
        #returns count extracts
        count_extracts = np.unique(extract, return_counts= True )
    else:
        count_extracts = np.unique(extract, return_counts= True )  

        #appends missing pos's
        fos = np.append(count_extracts[0],missing_pos)
        fos_zeros = np.append(count_extracts[1],missing_pos_zeros)
        fos_full = (fos,fos_zeros)

        count_extracts = fos_full

    return(count_extracts)

#turn it to dictionary, because I want to practice working with those
def dict_it(count_extracts):

    """A function for turning extracted ParOFSpeech values into a dictionary format."""

    dict_extracts = {}
    dict_list =[]
    for i in range(len(count_extracts[1])):
        if i == 0:
            dict_extracts["pos"] = count_extracts[0][0]
            dict_extracts["count"] = count_extracts[1][0]
            dict_list.append(dict_extracts)
        else:
            upd_ext = {"pos":count_extracts[0][i],"count":count_extracts[1][i]}
            dict_list.append(upd_ext)
    return(dict_list)


#in case the POS is not in the text, it would return 
#an array with one less item and that could scramble things
#so in case the asked POS is not there this part adds it as 0
def addmissing(extract,list_of_stuff):

    """ A function to fill in missing gaps in the PartOfSpeech values.
    If values are set as [NOUN,VERB,ADV] 
    output of [NOUN,VERB],[12,3] will become [NOUN,VERB,ADV],[12,3,0]"""

    missing_stuff = []
    for i in list_of_stuff:
        if i not in extract:
            missing_stuff.append(i)
        else:
            pass
    #making an equal length list of 0s
    missing_stuff_zeros = []
    for i in missing_stuff:
        missing_stuff_zeros.append(0)
    
    return(missing_stuff,missing_stuff_zeros)

# multiple files from 1 folder
def count_mfof(fp, ext = ["NOUN","VERB","ADJ","ADV"], el =["PERSON","GPE","ORG"], cm ='<.*?>'):

    """ A function for counting PartsOfSpeech in multiple files of a single folder.
    fp = filepath of the folder with multiple files.
    ext = list of PartOfSpeech values in spacy.pos_ syntax,
        defaults are ["NOUN","VERB","ADJ","ADV"]
    el = list of Named entity values in spacy.ent.label_ syntax,
        defaults are ["PERSON","GPE","ORG"]
    cm = regex specification to remove metadata
        defaults are '<.*?>' 
    Relies on additional functions:
     - dict_it(count_extracts)
     - count_ofof(...)
     - addmissing(extract,list_of_stuff)"""

    fp_s = sorted(os.listdir(fp))
    dict_list_list = []
    #get count out of the file
    for i in fp_s: 
        extract_array = count_ofof(fp = fp + i, extract_this = ext, ent_list = el, clean_metadata = cm)
        extract_dict = dict_it(extract_array)
        #add filename to each dict in list to prevent confusion on where the data came from
        for j in extract_dict:
            j.update({"file":i})
    
        dict_list_list.append(extract_dict)
    return(dict_list_list)    

#extract multiple files from multiple folders
def count_mfmf(fp,ext = ["NOUN","VERB","ADJ","ADV"], el =["PERSON","GPE","ORG"], cm ='<.*?>'):

    """ A function for counting PartsOfSpeech in multiple files of multipe folders.
    fp = filepath of the folder with multiple files.
    ext = list of PartOfSpeech values in spacy.pos_ syntax,
        defaults are ["NOUN","VERB","ADJ","ADV"]
    el = list of Named entity values in spacy.ent.label_ syntax,
        defaults are ["PERSON","GPE","ORG"]
    cm = regex specification to remove metadata
        defaults are '<.*?>' 
    Relies on additional functions:
     - dict_it(count_extracts)
     - count_ofof(...)
     - count_mfof(...)
     - addmissing(extract,list_of_stuff)"""

    #sort filenames
    fp_s = sorted(os.listdir(fp))
    #empty list for storing results
    folder_list = []
    #get count out of the folders
    for i in fp_s: 
        #use the many file one folder function to extract counts
        extract_array = count_mfof(fp = fp + i + "/", ext =ext, el = el, cm=cm)
        
        #to avoid confusion folder name will also be inserted into the dictionaries
        #for all files 
        for f in range(0,len(extract_array)):
            #into each dictionary of each folder of each file insert folder name
            for j in extract_array[f]:
                j.update({"folder" : i })

        #add results to list
        folder_list.append(extract_array)
        
    return(folder_list)    

#for formatting to wanted table structure
def table_format(one_file):

    """ formatting dictionaries into pandas.DataFrames according to assignment 1 specification."""
    
    df = pd.DataFrame(one_file)
    #pivot to wanted form
    df = df.pivot(index="file", columns = "pos", values = "count")
    #reorder to wanted form
    df = df[["NOUN","VERB","ADJ","ADV","PERSON","GPE","ORG"]]
    #make calculations
    df[["NOUN","VERB","ADJ","ADV"]] = df[["NOUN","VERB","ADJ","ADV"]].div(10000)
    #rename to wanted form
    df = df.rename(columns={"NOUN" : "RelFreq NOUN",
    "VERB": "RelFreq VERB",
    "ADJ" : "RelFreq ADJ",
    "ADV" : "RelFreq ADV",
    "PERSON" : "No. Unique PER",
    "GPE" : "No. Unique LOC",
    "ORG" : "No. Unique ORG",})
    #make it nice finishing touches, remove pos index, add filename
    df = df.reset_index()
    df = df.rename(columns={"file" : "Filename"})
    df = df.rename_axis(None, axis=1)

    return(df)