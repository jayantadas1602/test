# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/core/actions/#custom-actions/


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

import spacy
import os
from inspect import signature
import numpy as np


from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor 
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_absolute_error 

import logging

logger = logging.Logger(__name__)

nlp = spacy.load('en_core_web_lg')

Global_X = {"titanic":['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'], "heart":['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'], "students":['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p',	'salary', 'gender', 'ssc_b', 'hsc_b', 'degree_t', 'workex', 'specialisation'], "stroke":['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'],"housing price":['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']}
Global_Y = {"titanic":['Survived'], "heart":['target'], "students":['status'], "stroke":['stroke'],"housing price":["SalePrice"]}
DATASET = ''

def get_num_columns_reply(df):
    return "The {dataset} dataset contains "+get_num_columns(df)+" columns."
def get_num_columns(df):
    return str(df.shape[1])

def get_num_rows_reply(df):
    return "The {dataset} dataset contains "+get_num_rows(df)+" rows."
def get_num_rows(df):
    return str(df.shape[0])

def get_categorical_columns_reply(df):
    cols = get_categorical_columns(df)
    ans = "The {dataset} dataset contains "
    for c in cols:
        ans += (c + ', ')
    ans += "as Categorical columns"
    return ans
def get_categorical_columns(df):
    cols = df.columns
    num_cols = df._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))
    return cat_cols

def get_numerical_columns_reply(df):
    cols = get_numerical_columns(df)
    ans = "The {dataset} dataset contains "
    for c in cols:
        ans += (c + ', ')
    ans += "as Numerical columns"
    return ans
def get_numerical_columns(df):
    cols = df.columns
    num_cols = df._get_numeric_data().columns
    return num_cols

def check_null_values_reply(df):
    ans = check_null_values(df)
    if ans:
        return "Yes, there are some null values in {dataset} dataset"
    return "No, there is no null value found in {dataset} dataset"
def check_null_values(df):
    return df.isnull().any().any()

def remove_null_values_reply(df, file_path):
    df = remove_null_values(df)
    df.to_csv(file_path, index=False)
    return "The {dataset} dataset file is modified and null values are removed from it."
def remove_null_values(df):
    return df.dropna()

def descdata(df):
    return df.describe()

def replace_null_with_mean_reply(df, file_path):
    df = replace_null_with_mean(df)
    df.to_csv(file_path, index=False)
    return "The null values in the {dataset} dataset is replaced with mean of that respective column. You can see the changes in the dataset file."
def replace_null_with_mean(df):
    n_cols = get_numerical_columns(df)
    for col in n_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True)
    return df

def replace_null_with_median_reply(df, file_path):
    df = replace_null_with_median(df)
    df.to_csv(file_path, index=False)
    return "The null values in the {dataset} dataset is replaced with median of that respective column. You can see the changes in the dataset file."
def replace_null_with_median(df):
    n_cols = get_numerical_columns(df)
    for col in n_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    return df

def replace_null_with_mode_reply(df, file_path):
    df = replace_null_with_mode(df)
    df.to_csv(file_path, index=False)
    return "The null values in the {dataset} dataset is replaced with mode of that respective column. You can see the changes in the dataset file."
def replace_null_with_mode(df):
    n_cols = get_numerical_columns(df)
    for col in n_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode(), inplace=True)
    return df

def label_encode_categorical_columns_reply(df, file_path):
    df = label_encode_categorical_columns(df)
    df.to_csv(file_path, index=False)
    return "The {dataset} dataset file is modified to Label Encode Categorical columns."
def label_encode_categorical_columns(df):
    c_cols = get_categorical_columns(df)
    for col in c_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def standardize_numerical_columns_reply(df, file_path):
    df = standardize_numerical_columns(df)
    df.to_csv(file_path, index=False)
    return "The {dataset} dataset file is modified to Standardize Numerical columns."
def standardize_numerical_columns(df):
    n_cols = get_numerical_columns(df)
    scaler = MinMaxScaler()
    df[n_cols] = scaler.fit_transform(df[n_cols])
    return df

def bxplt(df):
    global Global_X
    global DATASET
    n_cols = Global_X[DATASET]
    #print(n_cols)
    #boxplot = df.boxplot(column=n_cols).get_figure()
    #np.random.seed(1234)
    #df = pd.DataFrame(np.random.randn(10, 4),
    #              columns=['Col1', 'Col2', 'Col3', 'Col4'])
    boxplot = df.boxplot(column=n_cols).get_figure()
    boxplot.savefig('box.png')
    return "The graph is stored in box.png"

def prpt(df):
    global Global_X
    global Global_Y
    global DATASET
    n_cols = Global_X[DATASET]
    m_cols = Global_Y[DATASET]
    # Set the figure's size
    plt.figure(figsize=(20,15))
    # Plot heatmap
    ppt=sb.heatmap(df.corr(), annot = True, cmap = 'Blues_r').get_figure().savefig('heatmap.png')
    #ppt.savefig('pairplot.png')
    return "The graph is stored in heatmap.png"


def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    
    plt.suptitle('Scatter and Density Plot')
    plt.savefig('scattermatrix.png')

def scpt(df):
    plotScatterMatrix(df, 20, 10)
    return "The graph is stored in scattermatrix.png"

def find_data_file_path(text, base_path='datasets/'):
    li = os.listdir(base_path)
    lis = []
    for l in li:
        a = l.split('.csv')[0]
        a = a.replace('_', ' ')
        lis.append(a)
    mapping = []
    for l in lis:
        t1 = nlp(l)
        t2 = nlp(text)
        mapping.append((t1.similarity(t2), l))
    mapping = sorted(mapping, reverse=True)
    file_name = mapping[0][1].replace(' ','_')+'.csv'
    return base_path+file_name

def get_data(file_path):
    df = pd.read_csv(file_path)
    return df

def ranfor(df):
    global Global_X
    global Global_Y
    global DATASET
    model=RandomForestClassifier(n_estimators=100)
    X_col=Global_X[DATASET]
    Y_col=Global_Y[DATASET]
    model.fit(df[X_col],df[Y_col])
    prediction7=model.predict(df[X_col])
    #return "The accuracy of the Random Forests is : " + str(metrics.accuracy_score(prediction7,df[Y_col]))
    return "The model is ready to be evaluated"
    
def logReg(df):
    global Global_X
    global Global_Y
    global DATASET
    model = LogisticRegression()
    X_col=Global_X[DATASET]
    Y_col=Global_Y[DATASET]
    logger.debug("Some log message")

    print(X_col," \n", Y_col)
    model.fit(df[X_col],df[Y_col])
    prediction3=model.predict(df[X_col])
    print('prediction3 ', prediction3)
    #return "The accuracy of the Logistic Regression is : " + str(metrics.accuracy_score(prediction3,df[Y_col]))
    return "The model is ready to be evaluated"

'''def dectree(df):
    global Global_X
    global Global_Y
    global DATASET
    model=DecisionTreeClassifier()
    X_col=Global_X[DATASET]   
    Y_col=Global_Y[DATASET]
    model.fit(df[X_col],df[Y_col])
    prediction4=model.predict(df[X_col])
    #return "The accuracy of the Decision Tree is : " + str(metrics.accuracy_score(prediction4,df[Y_col]))
    return "The model is ready to be evaluated"
'''

def knn(df):
    global Global_X
    global Global_Y
    global DATASET
    model=KNeighborsClassifier()
    X_col=Global_X[DATASET]   
    Y_col=Global_Y[DATASET]
    model.fit(df[X_col],df[Y_col])
    prediction5=model.predict(df[X_col])
    #return "The accuracy of the KNN is : " + str(metrics.accuracy_score(prediction5,df[Y_col]))
    return "The model is ready to be evaluated"

def rsgvm(df):
    global Global_X
    global Global_Y
    global DATASET
    X_col=Global_X[DATASET]
    Y_col=Global_Y[DATASET]
    clf = svm.SVR()
    clf.fit(df[X_col],df[Y_col]) 
    val_pred = clf.predict(df[X_col])
    return "The mean absoulute error of the SVM regressor is : " + str(metrics.mean_absolute_error(val_pred,df[Y_col]))
    
def dtg(df):
    global Global_X
    global Global_Y
    global DATASET
    X_col=Global_X[DATASET]
    Y_col=Global_Y[DATASET]
    iowa_model = DecisionTreeRegressor(random_state=1,max_leaf_nodes=100)
    iowa_model.fit(df[X_col],df[Y_col]) 
    val_pred = iowa_model.predict(df[X_col])
    return "The mean absoulute error of the Decision Tree Regressor is : " + str(metrics.mean_absolute_error(val_pred,df[Y_col]))

def accr_rf(df):
    global Global_X
    global Global_Y
    global DATASET
    model=RandomForestClassifier(n_estimators=100)
    X_col=Global_X[DATASET]
    Y_col=Global_Y[DATASET]
    model.fit(df[X_col],df[Y_col])
    prediction7=model.predict(df[X_col])
    return "The accuracy of the Random Forests is : " + str(metrics.accuracy_score(prediction7,df[Y_col]))

def f_scr_rf(df):
    global Global_X
    global Global_Y
    global DATASET
    model=RandomForestClassifier(n_estimators=100)
    X_col=Global_X[DATASET]
    Y_col=Global_Y[DATASET]
    model.fit(df[X_col],df[Y_col])
    prediction7=model.predict(df[X_col])
    return "The f1 score of the Random Forests is : " + str(metrics.f1_score(prediction7,df[Y_col]))

def lgls_rf(df):
    global Global_X
    global Global_Y
    global DATASET
    model=RandomForestClassifier(n_estimators=100)
    X_col=Global_X[DATASET]
    Y_col=Global_Y[DATASET]
    model.fit(df[X_col],df[Y_col])
    prediction7=model.predict(df[X_col])
    return "The log loss of the Random Forests is : " + str(metrics.log_loss(prediction7,df[Y_col]))


def get_reply_from_context(df, file_path, qtype, dataset):
    global DATASET
    DATASET = dataset.lower()

    function_sim = []
    for fm in function_map:
        t1 = nlp(fm[0])
        t2 = nlp(qtype)
        function_sim.append((t1.similarity(t2), fm[1]))
    function_sim = sorted(function_sim, reverse=True)
    fn = function_sim[0][1]
    sig = signature(fn)
    if len(sig.parameters) == 2:
        reply = fn(df, file_path).replace("{dataset}", dataset)
    else:
        reply = fn(df).replace("{dataset}", dataset)
    return reply


function_map = [
    ('columns', get_num_columns_reply),
    ('rows', get_num_rows_reply),
    ('Describe',descdata),
    ('check null values', check_null_values_reply),
    ('remove null values', remove_null_values_reply),
    ('replace null values with mean', replace_null_with_mean_reply),
    ('replace null values with median',replace_null_with_median_reply),
    ('replace null values with mode',replace_null_with_mode_reply),
    ('categorical columns', get_categorical_columns_reply),
    ('numerical columns', get_numerical_columns_reply),
    ('Label encode categorical columns', label_encode_categorical_columns_reply),
    ('standardize numerical columns', standardize_numerical_columns_reply),
    ('Train Random forest', ranfor),
    ('Accuracy of Random forest',accr_rf),
    ('f1 score of Random forest',f_scr_rf),
    ('log loss of Random forest',lgls_rf),
    ('Train Logistic regression', logReg),
    ('Train KNN', knn),
    ('Train Decision Tree Regression', dtg),
    ('Train SVM regression',rsgvm),
    ('Create boxplot',bxplt),
    ('Draw heatmap',prpt),
    ('Plot scattermatrix',scpt)
]

class ActionAnswer(Action):

    def name(self) -> Text:
        return "action_answer"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        qtype = tracker.get_slot("question_type")
        dataset = tracker.get_slot("dataset")
        file_path = find_data_file_path(dataset)
        df = get_data(file_path)
        #if qtype == 'Train':
                
        reply = get_reply_from_context(df, file_path, qtype, dataset)
        dispatcher.utter_message(text=reply)

        return [SlotSet("dataset", dataset)]
