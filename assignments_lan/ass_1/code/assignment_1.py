######################### import packages
#os to fast track terminal use
import os
os.system("pip install pandas spacy")
#spacy for nlp
import spacy
os.system("python -m spacy download en_core_web_md")
nlp = spacy.load('en_core_web_md')
#pandas for dataframes
import pandas as pd
#numpy for hjelp
import numpy as np
#for cleaning data
import re
#custom functions
import b_func as bf

############################# CODE
#set input filepath
fp = os.path.join("..","in")
fp = "/work/lang_anal/CDS_lang/assignments_lan/ass_1/in/"

#get all files and count POSs
c_pos = bf.count_mfmf(fp)

#Format values to table for all files in a table, and saving in separate csv files
#for all folders
for j in range(0,len(c_pos)):
    cter = len(j)
    #for all files
    for i in c_pos[j]:
        if cter == len(j):
            df_perm = bf.table_format(i)
        elif cter > 1 :
            df_temp = bf.table_format(i)
            df_perm = pd.concat([df_perm,df_temp], ignore_index = True)
        elif cter == 1:
            df_temp = bf.table_format(i)
            df_perm = pd.concat([df_perm,df_temp], ignore_index = True)

            ##the last iteration for the folder saves the df
            ##folder name is saved in the dictionary of data, it is extracted
            folder = i[0].get("folder")
            filename = os.path.join("..","out",folder)

            df_perm.to_csv(filename + ".csv", sep=',', index=False, encoding='utf-8')
        
        cter = cter-1

