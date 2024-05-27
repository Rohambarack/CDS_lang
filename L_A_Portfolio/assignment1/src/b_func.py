
import os
import spacy
nlp = spacy.load('en_core_web_md')
import pandas as pd
import numpy as np
import re

# function for 1 file from 1 folder
def count_one_file_one_folder(fp, extract_this = ["NOUN","VERB","ADJ","ADV"], ent_list = ["PERSON","GPE","ORG"], clean_metadata = '<.*?>'):

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
    text_clean = re.sub(clean_metadata, '', text)

    #convert it with spacy
    text_nlp = nlp(text_clean)

    #extract values
    #empty list for extracts
    extract_pos = []
    for part_of_speech in text_nlp:
        if part_of_speech.pos_ in extract_this:
            extract_pos.append(part_of_speech.pos_)
        else:
            pass 
    
    #get named entities out
    extract_named_entity = []
    for named_entity in text_nlp.ents:
        if named_entity.label_ in ent_list:
            extract_named_entity.append(named_entity.label_)
        else:
            pass
    
    #add lists if there were none
    missing_pos, missing_pos_zeros = addmissing(extract_pos,extract_this)
    missing_ents, missing_ents_zeros = addmissing(extract_named_entity, ent_list)

    #regroup
    extract = extract_pos + extract_named_entity
    missing_counts = missing_pos + missing_ents
    missing_count_zeros = missing_pos_zeros + missing_ents_zeros

    #returning counts of unique occurences
    if len(missing_counts) == 0:
        #returns count extracts
        count_extracts = np.unique(extract, return_counts= True )
    else:
        count_extracts = np.unique(extract, return_counts= True )  

        #appends missing pos's
        parts_of_speech_and_named_entity = np.append(count_extracts[0],missing_counts)
        parts_of_speech_and_named_entity_zeros = np.append(count_extracts[1],missing_count_zeros)
        count_full = (parts_of_speech_and_named_entity,parts_of_speech_and_named_entity_zeros)

        count_extracts = count_full
    
    ## (v_0.2) fixing frequency count
    all_words_in_text = len(text_nlp)
    all_word_plus_tags = np.append(count_extracts[0], "N_WORDS")
    all_word_plus_counts = np.append(count_extracts[1], all_words_in_text)
    count_extracts = (all_word_plus_tags,all_word_plus_counts)

    return(count_extracts)

#turn it to dictionary, because I want to practice working with those
def make_it_a_dictionary(count_extracts):

    """A function for turning extracted ParOFSpeech values into a dictionary format."""

    dictionary_extracts = {}
    dictionary_list =[]
    for index_of_extract_in_array in range(len(count_extracts[1])):
        if index_of_extract_in_array == 0:
            dictionary_extracts["pos"] = count_extracts[0][0]
            dictionary_extracts["count"] = count_extracts[1][0]
            dictionary_list.append(dictionary_extracts)
        else:
            upd_ext = {"pos":count_extracts[0][index_of_extract_in_array],"count":count_extracts[1][index_of_extract_in_array]}
            dictionary_list.append(upd_ext)
    return(dictionary_list)


#in case the POS is not in the text, it would return 
#an array with one less item and that could scramble things
#so in case the asked POS is not there this part adds it as 0
def addmissing(extract,list_of_stuff):

    """ A function to fill in missing gaps in the PartOfSpeech values.
    If values are set as [NOUN,VERB,ADV] 
    output of [NOUN,VERB],[12,3] will become [NOUN,VERB,ADV],[12,3,0]"""

    missing_stuff = []
    for individual_elements in list_of_stuff:
        if individual_elements not in extract:
            missing_stuff.append(individual_elements)
        else:
            pass
    #making an equal length list of 0s
    missing_stuff_zeros = []
    for individual_elements in missing_stuff:
        missing_stuff_zeros.append(0)
    
    return(missing_stuff,missing_stuff_zeros)

# multiple files from 1 folder
def count_many_files_one_folder(fp, ext = ["NOUN","VERB","ADJ","ADV"], el =["PERSON","GPE","ORG"], cm ='<.*?>'):

    """ A function for counting PartsOfSpeech in multiple files of a single folder.
    fp = filepath of the folder with multiple files.
    ext = list of PartOfSpeech values in spacy.pos_ syntax,
        defaults are ["NOUN","VERB","ADJ","ADV"]
    el = list of Named entity values in spacy.ent.label_ syntax,
        defaults are ["PERSON","GPE","ORG"]
    cm = regex specification to remove metadata
        defaults are '<.*?>' 
    Relies on additional functions:
     - make_it_a_dictionary(count_extracts)
     - count_one_file_one_folder(...)
     - addmissing(extract,list_of_stuff)"""

    filepath_sorted = sorted(os.listdir(fp))
    dict_list_list = []
    #get count out of the file
    for filename in filepath_sorted: 
        extract_array = count_one_file_one_folder(fp = fp + filename, extract_this = ext, ent_list = el, clean_metadata = cm)
        extract_dict = make_it_a_dictionary(extract_array)
        #add filename to each dict in list to prevent confusion on where the data came from
        for extracted_dictionaries in extract_dict:
            extracted_dictionaries.update({"file":filename})
    
        dict_list_list.append(extract_dict)
    return(dict_list_list)    

#extract multiple files from multiple folders
def count_many_files_many_folders(fp,ext = ["NOUN","VERB","ADJ","ADV"], el =["PERSON","GPE","ORG"], cm ='<.*?>'):

    """ A function for counting PartsOfSpeech in multiple files of multipe folders.
    fp = filepath of the folder with multiple files.
    ext = list of PartOfSpeech values in spacy.pos_ syntax,
        defaults are ["NOUN","VERB","ADJ","ADV"]
    el = list of Named entity values in spacy.ent.label_ syntax,
        defaults are ["PERSON","GPE","ORG"]
    cm = regex specification to remove metadata
        defaults are '<.*?>' 
    Relies on additional functions:
     - make_it_a_dictionary(count_extracts)
     - count_one_file_one_folder(...)
     - count_many_files_one_folder(...)
     - addmissing(extract,list_of_stuff)"""

    #sort filenames
    filepath_sorted = sorted(os.listdir(fp))
    #empty list for storing results
    folder_list = []
    #get count out of the folders
    for foldername in filepath_sorted: 
        #use the many file one folder function to extract counts
        extract_array = count_many_files_one_folder(fp = fp +  foldername + "/", ext =ext, el = el, cm=cm)
        
        #to avoid confusion folder name will also be inserted into the dictionaries
        #for all files 
        for index_of_folder in range(0,len(extract_array)):
            #into each dictionary of each folder of each file insert folder name
            for file_count_dictionary in extract_array[index_of_folder]:
                file_count_dictionary.update({"folder" : foldername })

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
    df = df[["NOUN","VERB","ADJ","ADV","PERSON","GPE","ORG","N_WORDS"]]
    #make calculations (v_0.2) adding all words in text and rounding freq to 3 decimals
    df[["NOUN","VERB","ADJ","ADV"]] = df[["NOUN","VERB","ADJ","ADV"]].div(df["N_WORDS"], axis=0)*10000
    df = df.round(3)
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