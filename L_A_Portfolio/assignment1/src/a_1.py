#import os + pd + custom functions
from codecarbon import EmissionsTracker
from codecarbon import track_emissions
import scipy
import os
import re
import b_func as bf
import pandas as pd
import matplotlib.pyplot as plt

def flatten(xss):
    #flatten outputs from the re.findall function
    return [x for xs in xss for x in xs]

def find_semester(column):
    #extract semester from filename
    mod_0_list = []
    for i in column:
        mod_0 = re.findall(r'\.([^\.]+)\.',i)
        mod_0_list.append(mod_0)

    mod_0_list = flatten(mod_0_list)

    mod_1_list = []
    for i in mod_0_list:
        mod_1 = re.findall(r'^.',i)
        mod_1_list.append(mod_1)
    
    mod_1_list = flatten(mod_1_list)

    return mod_1_list

def save_plots(var_name,df_large):
    # Converting to wide dataframe 
    data_wide = df_large.pivot_table(index="Filename", 
                                    columns="Semester", 
                                    values=var_name, 
                                    aggfunc='first') 
    
    
    # plotting multiple density plot 
    plot = data_wide.plot.kde(figsize = (5, 5), 
                    linewidth = 2,
                    )
    plot.set_title(var_name)
    output = os.path.join("..","plots",var_name + ".png")
    plt.savefig(output)

def analyse():
    """a function to make plots of the results for a short interpretation of them"""
    #find csvs
    filepath = os.path.join("..","out")
    filelist = sorted(os.listdir(filepath))
    #merge'em up
    df_list = []
    for i in filelist:
        path = os.path.join(filepath,i)
        df = pd.read_csv(path)
        df_list.append(df)


    df_large = pd.concat(df_list)
    #find semester
    semester = find_semester(df_large["Filename"])
    #add it into df
    df_large.insert(1, "Semester", semester, True)
    #getting variable names for plots
    inspect = df_large.columns[2:]

    #loop through variables and plot them
    for i in inspect:
        save_plots(i,df_large)

def main():
    #A5#####################################################
    a5_out = os.path.join("..","..","assignment5","out")
    tracker = EmissionsTracker(project_name="A1",
                            output_dir=a5_out,
                            output_file="emissions_a1.csv")

    ##################################################
    # tracking data downloading
    tracker.start_task("load_model")
    

    #set input filepath
    filepath = os.path.join("..","in/")
    #get all files and count POSs
    count_POS = bf.count_many_files_many_folders(filepath)

   
    a1_load_count = tracker.stop_task()
    ########################################
    # tracking downloading and initializing model
    tracker.start_task("analyse_save")

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

    #add some plots
    analyse()
    ###################################################
    a1_format_save = tracker.stop_task()
    _ = tracker.stop()

if __name__ == "__main__":
    main()