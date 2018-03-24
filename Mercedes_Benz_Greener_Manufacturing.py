import math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.linear_model.stochastic_gradient import SGDRegressor
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import scoreatpercentile
from sklearn.datasets.samples_generator import make_regression
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.linear_model.ridge import Ridge
from sklearn.utils import shuffle
from subprocess import check_output

###########################################################################

pd.options.mode.chained_assignment = None  
pd.options.display.max_columns = 999

#Preparing Training and Testing data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print("Train shape : ", train_df.shape)
print("Test shape : ", test_df.shape)

#Encoding Step (Method 1) 
for f in ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values)) 
        train_df[f] = lbl.transform(list(train_df[f].values))
     
train_y = train_df['y'].values
train_X = train_df.drop(["ID", "y"], axis=1)


from sklearn import ensemble
model = ensemble.RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)
model.fit(train_X, train_y)
feat_names = train_X.columns.values

# Importances 
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(importances)[::-1][:20]


 ############################################################################


df=pd.read_csv("train.csv",usecols=feat_names[indices])
df1=pd.read_csv("test.csv",usecols=feat_names[indices])

# Encoding Step (Method 2)
for c in df.columns:
    if df[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(df[c].values) + list(df1[c].values))
        df[c] = lbl.transform(list(df[c].values))
        df1[c] = lbl.transform(list(df1[c].values)) 

#Preparing Training and Testing data
X=np.array(df)
y=np.array(train_df['y'])

#Regression Engine
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.01)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)


#Predicting and Saving Results
forecast_set = clf.predict(np.array(df1))
output = pd.DataFrame({'id': test_df['ID'].astype(np.int32), 'y': forecast_set})
output.to_csv('answers.csv', index=False)
