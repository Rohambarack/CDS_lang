#packages
# system tools
import os
#custom functions
import b_func as bf

# data  tools
import pandas as pd
import numpy as np

#a5
from codecarbon import EmissionsTracker
from codecarbon import track_emissions
#plots
import matplotlib.pyplot as plt

# Machine learning stuff
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

def main():
    #A5##########################################
    a5_out = os.path.join("..","..","assignment5","out")
    tracker = EmissionsTracker(project_name="A2_log",
                            output_dir=a5_out,
                            output_file="emissions_a2_log.csv")

    #############################################
    #load data
    tracker.start_task("load")
    # separate sets, vectorize
    fp = os.path.join("..","data","fake_or_real_news.csv")
    train_list, test_list = bf.preprocDF(fp, text="text",label="label",vectit= 0)

    a2_load_log = tracker.stop_task()
    #############################################
    #model, modify, operate on the data
    tracker.start_task("model")
    #vectorize data
    vectorisation = TfidfVectorizer(ngram_range=(1,2),
                            lowercase = True,
                            max_df = .95,
                            min_df = .05,
                            max_features = 500)
            
    #apply vectorizer
    x_train_features = vectorisation.fit_transform(train_list[0])
    x_test_features = vectorisation.transform(test_list[0])
    train_l = [x_train_features,train_list[1]]
    test_l = [x_test_features,test_list[1]]

    #define classifier
    classifier = LogisticRegression(random_state=42).fit(train_l[0], train_l[1])
    #predict
    prediction = classifier.predict(test_l[0])

    a2_model_log = tracker.stop_task()
    ###############################################
    #create outputs, analyze 
    tracker.start_task("analyse")
    #conf_matrix
    cm = metrics.confusion_matrix(test_l[1],prediction, labels=classifier.classes_)
    cm_disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                        display_labels=classifier.classes_)
    #report
    cr = metrics.classification_report(test_l[1], prediction)

    a2_analyse_log = tracker.stop_task()
    ################################################
    #save output
    tracker.start_task("save")
    save_path = os.path.join("..","out")
    #classification report
    f = open(os.path.join(save_path,"cr_log.txt"), 'w')
    f.write('Logistic Classifier output\n\nClassification Report\n\n{}'.format(cr))
    f.close()
    #confusion mtrix
    cm_disp.plot()
    plt.savefig(os.path.join(save_path,"cm_log.png"))

    from joblib import dump, load
    dump_path = os.path.join("..","models")
    dump(classifier, os.path.join(dump_path,"Log_reg.joblib"))
    dump(vectorisation, os.path.join(dump_path,"Vectorizer_Log_reg.joblib"))

    a2_save_log = tracker.stop_task()
    #################################################
    _ = tracker.stop()

if __name__ == "__main__":
    main()
