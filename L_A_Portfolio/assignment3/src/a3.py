import os
import re
import argparse
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import gensim
import gensim.downloader as api
from codecarbon import EmissionsTracker


def input_parser():
    parser =  argparse.ArgumentParser(description='Spotify assignment inputs')
    parser.add_argument('--artist',
                        "-a",
                        required = True,
                        help='Name of artist')
    parser.add_argument('--query_word',
                        "-qw",
                        required = True,
                        help='Word, which should be found in the discography')
    parser.add_argument('--data',
                        "-d",
                        default = "Spotify Million Song Dataset_exported.csv",
                        help='Dataset in the "in" folder. Default is the spotify dataset, data_spoty.csv')
    parser.add_argument('--n_embeds',
                        "-n_e",
                        default = 5,
                        help='extends the query to these amount of closest terms')
    

    args = parser.parse_args()
    return args

def clean_text(text):
    clean_text = re.sub(r"""
                    [,.;@#?!&$\n]+  # Accept one or more copies of punctuation
                    \ *           # plus zero or more copies of a space,
                    """,
                    " ",          # and replace it with a single space
                    text.lower(), flags=re.VERBOSE)
    return clean_text


def non_zero_counter(list_of_values):
    non_zero_counter = 0
    for count in list_of_values:
        if type(count) != str: #the first is the title so that is a str

            if count > 0:
                    non_zero_counter += 1
        else:
            pass
    else:
        pass

    return non_zero_counter

def make_output_tables(query_result, max_row):
    #make tables
    #extended table
    ex_data = pd.DataFrame(query_result["song"])
    ex_data
    #small info table
    small_df = pd.DataFrame(columns=["Artist","Term","Percentage","All_Percentage","N_Songs"])
    #add terms
    small_df.Term = ex_data.columns[1:len(ex_data.columns)]
    #add artist
    small_df.Artist = ex_data.columns[0] 
    #add n_songs
    small_df.N_Songs = max_row
    #add all_percentage
    small_df.All_Percentage = round(len(ex_data)/max_row*100,2)
    #add percentages for each query term
    #look through all colums of the terms in the big set, find non zeros, make it a list
    ex_data.fillna(0)
    percentages = []
    for col in ex_data.columns[1:len(ex_data.columns)]:
        n_z_count = non_zero_counter(ex_data[col])
        percentages.append(round(n_z_count/max_row*100,2))

    small_df.Percentage = percentages

    return ex_data, small_df

class Extended_Query:

    def __init__(self, data, model, artist, q_word, n_embeds):
        self.data = data
        self.model = model
        self.n_embeds = n_embeds
        self.artist = artist
        self.q_word = q_word
    
    def create_query(self):
        """ Method for creating a query from model and data,
        based on artist and query extension size
        - data: csv file with columns: 'artist', title as 'song', lyrics as 'text
        - model: word embedding model
        - artist: name of artist
        - q_word: lowercase str without punctuations.
        - n_embed: additional query extensions based on proximity in the embedding model"""
        #subset
        singer = self.artist
        sub_data = self.data.loc[self.data['artist'] == singer]
        #embed
        query_word = self.q_word
        n_similar = self.n_embeds
        query_array = self.model.most_similar(query_word,topn=n_similar)
        query_array.append((query_word, 1))

        # find in subset (subset rows by indexing row number to sub_data.loc)
        max_row = len(sub_data.index)
        occurence_count_full = {"song":[]}
        for row_index in range(0,max_row):
            row = sub_data.loc[row_index]
            song_title = row["song"]
            song_lyric = row["text"]
            artist = row["artist"]
            #clean lyrics to be compatible with the word embedding labels
            clean_lyric = clean_text(song_lyric)
            #for each word in list
            occurence_count = {}
            for word_value_pair in query_array:
                target_word = word_value_pair[0]
                #count list elements in lyric
                n_occurences = clean_lyric.split().count(target_word)
                temporary_dict = {artist:song_title,target_word:n_occurences}
                #check counts in the dictionary. if none found, it isn't saved
                n_z_counter = non_zero_counter(temporary_dict.values())

                if n_z_counter > 0:
                    occurence_count.update(temporary_dict)
                else:
                    pass

            #append dict if not empty
            if len(occurence_count) > 0:     
                occurence_count_full["song"].append(occurence_count)
            else:
                pass
        return occurence_count_full, max_row

def find_small_df():
    """ A function for extracting results for revisualizing """
    #get filepath
    filepath = os.path.join("..","out")
    files = os.listdir(filepath)
    #get filename
    #it starts with small, regardless of Artist
    for filename in files:
        find_small = filename.find("small")
        if (find_small != -1):
            correct_file = filename
        else:
            pass
    #full path
    total_filepath = os.path.join(filepath,correct_file)
    #read_csv
    artist_df = pd.read_csv(total_filepath)

    return artist_df

def plot_song_results(artist_df,query):
    """ a function for turning .csv outputs into a nice, interpretable plot"""
    df = artist_df.copy(deep=True)
    filtered = df.filter(items=["Term","Percentage"])
    q_term = query
    p_songs = artist_df["All_Percentage"][0]

    n_songs = artist_df["N_Songs"][0]
    filtered.plot.bar(x= "Term")
    plt.title("{}".format(artist_df["Artist"][0]))
    plt.legend([" # songs with {} related terms: {}\n % of all songs with {} related terms: {}".format(q_term,
                                                                                                        n_songs,
                                                                                                        q_term,
                                                                                                        p_songs)])
    return(plt.gcf())            

def main():
    #A5#############################################
    a5_out = os.path.join("..","..","assignment5","out")
    tracker = EmissionsTracker(project_name="A3",
                            output_dir=a5_out,
                            output_file="emissions_a3.csv")
    #################################################
    a3_load = tracker.start_task("load")

    args = input_parser()
    model = api.load("glove-wiki-gigaword-50")
    #data
    data_name = args.data
    data_path = os.path.join("..","in",data_name)
    data = pd.read_csv(data_path)

    tracker.stop_task()
    ########################################################
    a3_model = tracker.start_task("model")

    query_class = Extended_Query(model = model,
                                 data = data,
                                 artist = args.artist,
                                 q_word = args.query_word,
                                 n_embeds = args.n_embeds)
    
    query, n_songs = query_class.create_query()

    tracker.stop_task()
    ########################################################
    a3_analyse = tracker.start_task("analyse")

    extendend_table, small_table  = make_output_tables(query, n_songs)

    nice_plot = plot_song_results(small_table,args.query_word)

    tracker.stop_task()
    ########################################################
    a3_save = tracker.start_task("save")
    #save output
    output_path = os.path.join("..","out")
    filename_1 = "extended_query_"+ args.artist + ".csv"
    extendend_table.to_csv(os.path.join(output_path,filename_1), index=True)

    filename_2 = "small_table_"+ args.artist + ".csv"
    small_table.to_csv(os.path.join(output_path,filename_2), index=False)
    
    #saveplot
    filename_3 = "plot_" + args.artist + ".png"
    nice_plot.savefig(os.path.join(output_path,filename_3))
    

    tracker.stop_task()
    #######################################################
    _ = tracker.stop()

if __name__ == "__main__":
    main()