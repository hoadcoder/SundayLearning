#!/usr/bin/python

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from tools.utils import create_ratio_column, train_and_get_metrics

print("Loading clean data.")
df = pd.read_pickle("clean_data.pkl")

# Class Column
class_column = ["poi"]

# Input Columns
# Email
from_columns = ["from_poi_to_this_person", "from_messages"]
to_columns = ["from_this_person_to_poi", "to_messages"]
shared_columns = ["shared_receipt_with_poi"]
email_columns = from_columns + to_columns + shared_columns

# Payment
payment_columns = [
    "salary", 
    "bonus", 
    "long_term_incentive", 
    "deferred_income", 
    "deferral_payments", 
    "loan_advances", 
    "other", 
    "expenses", 
    "director_fees", 
    "total_payments"
]

# Stock
stock_columns = [
    "exercised_stock_options", 
    "restricted_stock", 
    "restricted_stock_deferred", 
    "total_stock_value"
]

# Other
other_columns = list(filter(
    lambda col: col not in payment_columns + stock_columns + email_columns + class_column, 
    df.columns
))

print("Create ratio columns")
create_ratio_column(df, payment_columns[-1], *payment_columns[:-1])
create_ratio_column(df, stock_columns[-1], *stock_columns[:-1])
create_ratio_column(df, from_columns[-1], *from_columns[:-1])
create_ratio_column(df, to_columns[-1], *to_columns[:-1])


pre_ml_df = df.copy()

print("Splitting X from y")
X = pre_ml_df.drop(class_column, axis=1).to_numpy()
y = pre_ml_df[class_column].to_numpy().ravel()

print("Done")

print("Rescaling features")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("Feature Selection")
kbest = SelectKBest(k="all")
kbest.fit(X_scaled, y)

scores_df = pd.DataFrame({
    "column_names": df.drop("poi", axis=1).columns,
    "scores": kbest.scores_
}).sort_values(by="scores", ascending=False)

cut_column = "salary"
cut_value = scores_df[scores_df["column_names"] == "expenses"]["scores"].values[0]

allowed_columns = scores_df[scores_df["scores"] >= cut_value]

index_to_select = list(allowed_columns.index)


clfs = [
    DecisionTreeClassifier, 
    RandomForestClassifier,
    LogisticRegression,
    SVC,
    GaussianNB,
    KNeighborsClassifier,
]

cvs = [3, False]
index_selections = [False, index_to_select]

#4 Rounds de iteração:
#    1 Round: Com CV de 3 folds e sem filtrar colunas
#    2 Round: Sem CV e sem filtrar colunas
#    
#    3 Round: Com CV de 3 folds e filtrando colunas
#    2 Round: Sem CV e filtrando colunas


dfs = []
for clf in clfs:
    for cv in cvs:
        for index_selection in index_selections:
            metrics = train_and_get_metrics(
                    X_scaled, y, 
                    classifier=clf(), 
                    cv=cv,
                    select_index=index_selection)
            
            df_i = pd.DataFrame(metrics)
            
            df_i["cross_val"] = cv > 1
            df_i["kbest"] = type(index_selection) == list
            
            dfs.append(df_i)
            
metrics_df = pd.concat(dfs)
            
            

for clf in 
for cv in [3, False]:

print("Trial with Cross Validation")
print("Decision Tree")
dt = DecisionTreeClassifier()
dt_metrics = train_and_get_metrics(X_scaled, y, dt, select_index=index_to_select)

print("Random Forest")
rf = RandomForestClassifier()
rf_metrics = train_and_get_metrics(X_scaled, y, rf, select_index=index_to_select)

print("Logistic Regression")
lr = LogisticRegression()
lr_metrics = train_and_get_metrics(X_scaled, y, lr, select_index=index_to_select)

print("Support Vector Machine")
svc = SVC()
svc_metrics = train_and_get_metrics(X_scaled, y, svc, select_index=index_to_select)

print("Naive Bayes")
nb = GaussianNB()
nb_metrics = train_and_get_metrics(X_scaled, y, nb, select_index=index_to_select)

print("K Nearest Neighbors")
knn = KNeighborsClassifier()
knn_metrics = train_and_get_metrics(X_scaled, y, knn, select_index=index_to_select)



print("Trial withOUT Cross Validation")
print("Decision Tree")
dt = DecisionTreeClassifier()
dt_metrics_nocv = train_and_get_metrics(X_scaled, y, dt, cv=False, select_index=index_to_select)

print("Random Forest")
rf = RandomForestClassifier()
rf_metrics_nocv = train_and_get_metrics(X_scaled, y, rf, cv=False, select_index=index_to_select)

print("Logistic Regression")
lr = LogisticRegression()
lr_metrics_nocv = train_and_get_metrics(X_scaled, y, lr, cv=False, select_index=index_to_select)

print("Support Vector Machine")
svc = SVC()
svc_metrics_nocv = train_and_get_metrics(X_scaled, y, svc, cv=False, select_index=index_to_select)

print("Naive Bayes")
nb = GaussianNB()
nb_metrics_nocv = train_and_get_metrics(X_scaled, y, nb, cv=False, select_index=index_to_select)

print("K Nearest Neighbors")
knn = KNeighborsClassifier()
knn_metrics_nocv = train_and_get_metrics(X_scaled, y, knn, cv=False, select_index=index_to_select)


metrics_dicts = [
    dt_metrics,
    rf_metrics,
    lr_metrics,
    svc_metrics,
    nb_metrics,
    knn_metrics,
    dt_metrics_nocv,
    rf_metrics_nocv,
    lr_metrics_nocv,
    svc_metrics_nocv,
    nb_metrics_nocv,
    knn_metrics_nocv,
]


dfs = []
for metrics_dict in metrics_dicts:
    df_i = pd.DataFrame(metrics_dict)
    
    if len(df_i) > 1:
        df_i["cross_val"] = True
        
    else:
        df_i["cross_val"] = False
        
    dfs.append(df_i)
    
metrics_df = pd.concat(dfs)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

