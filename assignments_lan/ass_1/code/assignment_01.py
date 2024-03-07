#import os + pd + custom functions
import os
import b_func as bf
import pandas as pd

#set input filepath
filepath = os.path.join("..","in/")
#get all files and count POSs
count_POS = bf.count_many_files_many_folders(filepath)

#Format values to table for all files in a table, and saving in separate csv files
#for all folders
for folder_list_index in range(0,len(count_POS)):
    counter = len(count_POS[folder_list_index])
    #for all files in a folder
    for file in count_POS[folder_list_index]:
        if counter == len(count_POS[folder_list_index]):
            df_perm = bf.table_format(file)
        elif counter > 1 :
            df_temp = bf.table_format(file)
            df_perm = pd.concat([df_perm,df_temp], ignore_index = True)
        elif counter == 1:
            df_temp = bf.table_format(file)
            df_perm = pd.concat([df_perm,df_temp], ignore_index = True)

            ##the last iteration for the folder saves the df
            ##folder name is saved in the dictionary of data, it is extracted
            folder = file[0].get("folder")
            filename = os.path.join("..","out",folder)

            df_perm.to_csv(filename + ".csv", sep=',', index=False, encoding='utf-8')
        
        counter = counter-1