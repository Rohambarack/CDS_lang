import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    #read in all data
    #path
    data_path = os.path.join("..","out")
    #files
    files = sorted(os.listdir(data_path))
    #filter not good ones
    files = files[6:]
    df_list = []
    for i in files:
    
        df = pd.read_csv(os.path.join(data_path,i))
    
        df_list.append(df)

    df_big = pd.concat(df_list) 
    df = df_big[["project_name","task_name","emissions"]]

    def mean_it(df):
        df = df.groupby(["project_name","task_name"]).mean("emission").reset_index()
        return(df)

    #separate
    df_1 = mean_it(df[df["project_name"]=="A1"])
    df_2nn = mean_it(df[df["project_name"]=="A2_nn"])
    df_2lg = mean_it(df[df["project_name"]=="A2_log"])
    df_3 = mean_it(df[df["project_name"]=="A3"])
    df_4 = mean_it(df[df["project_name"]=="A4"])

    #plot
    df_1.plot.bar(x = "task_name")
    plt.suptitle("A1")
    plt.savefig(os.path.join("..","plots","a1.png"))
    plt.close()

    df_2nn.plot.bar(x = "task_name")
    plt.suptitle("A2_nn")
    plt.savefig(os.path.join("..","plots","a2nn.png"))
    plt.close()


    df_2lg.plot.bar(x = "task_name")
    plt.suptitle("A2_lg")
    plt.savefig(os.path.join("..","plots","a2lg.png"))
    plt.close()


    df_3.plot.bar(x = "task_name")
    plt.suptitle("A3")
    plt.savefig(os.path.join("..","plots","a3.png"))
    plt.close()


    df_4.plot.bar(x = "task_name")
    plt.suptitle("A4")
    plt.savefig(os.path.join("..","plots","a4.png"))
    plt.close()

    #Compare
    df_sum = pd.DataFrame([{"task": "A1", "emission": df_1["emissions"].sum()},
                        {"task": "A2nn", "emission": df_2nn["emissions"].sum()},
                        {"task": "A2lg", "emission": df_2lg["emissions"].sum()},
                        {"task": "A3", "emission": df_3["emissions"].sum()},
                        {"task": "A4", "emission": df_4["emissions"].sum()}])

    df_sum.plot.bar(x = "task")
    plt.savefig(os.path.join("..","plots","a_comp.png"))


if __name__ == "__main__":
    main()