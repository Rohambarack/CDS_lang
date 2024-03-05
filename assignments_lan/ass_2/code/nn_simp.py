#packages
# system tools
import os
os.system("pip install pandas scikit-learn matplotlib numpy")
#custom functions
import b_func as bf

# data  tools
import pandas as pd
import numpy as np

# Machine learning stuff
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

# separate sets, vectorize
fp = os.path.join("in","fake_or_real_news.csv")
train_l, test_l = bf.preprocDF_2(fp, text="text",label="label")

#define classifier
classifier = MLPClassifier(activation = "logistic",
                           hidden_layer_sizes = (20,),
                           max_iter=1000,
                           random_state = 42).fit(train_l[0],train_l[1])


#predict
prediction = classifier.predict(test_l[0])
#conf_matrix
cm = np.array2string(metrics.confusion_matrix(test_l[1],prediction))
#report
cr = metrics.classification_report(test_l[1], prediction)

#save output
f = open('out/nn_report.txt', 'w')
f.write('NN Classifier output\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(cr, cm))
f.close()

from joblib import dump, load
dump(classifier, "models/nn.joblib")
