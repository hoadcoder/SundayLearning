# -*- coding: utf-8 -*-
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import fbeta_score, precision_score, recall_score, accuracy_score



def create_ratio_column(df, denominator, *numerators):
    for numerator in numerators:
        ratio_column_name = numerator + "_ratio"
        ratio_series = df[numerator] / df[denominator]
        ratio_series = ratio_series.replace(np.inf, np.nan)\
            .replace(-np.inf, np.nan)\
            .fillna(0.)
            
        df[ratio_column_name] = ratio_series
        
        
def classify_and_predict(metrics, classifier, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
            
    y_pred = classifier.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prc = precision_score(y_test, y_pred)
    rcl = recall_score(y_test, y_pred)
    f2s = fbeta_score(y_test, y_pred, beta=2)
    
    metrics["classifier"].append(classifier)
    metrics["accuracy"].append(acc)
    metrics["precision"].append(prc)
    metrics["recall"].append(rcl)
    metrics["f2"].append(f2s)
        

def train_and_get_metrics(X, y, classifier, cv=3, select_index=False):
    metrics = {
        "classifier": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f2": []
    }
    
    
    if select_index:
        X = X[:, select_index]

    
    if cv:
        kfold = StratifiedKFold(cv, random_state=42)
        for train_idx, val_idx in kfold.split(X, y):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
            
            classify_and_predict(metrics, classifier, X_train, y_train, X_val, y_val)
            
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42)

        classify_and_predict(metrics, classifier, X_train, y_train, X_test, y_test)
        
    return metrics
        
        
    
def cross_val_and_train(X, y, classifier):
    kfold = StratifiedKFold(3)
    
    metrics = {
        "classifier": classifier,
        "precision": [],
        "recall": [],
        "f2_score": [],
    }
    
    for train_idx, val_idx in kfold.split(X, y):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        
        classifier.fit(X_train, y_train)
        
        y_pred = classifier.predict(X_val)
        
        metrics["precision"].append(precision_score(y_val, y_pred))
        metrics["recall"].append(recall_score(y_val, y_pred))
        metrics["f2_score"].append(fbeta_score(y_val, y_pred, beta=2))

    return metrics


def no_cross_val_and_train(X, y, classifier):
    
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, stratify=y, random_state=42)

    classifier.fit(X_train, y_train)

    # Metrics for Test    
    y_pred = classifier.predict(X_test)
    test_prc = precision_score(y_test, y_pred)
    test_rcl = recall_score(y_test, y_pred)
    test_f2 = fbeta_score(y_test, y_pred, beta=2)

    # Metrics for Train
    y_pred = classifier.predict(X_train)
    train_prc = precision_score(y_train, y_pred)
    train_rcl = recall_score(y_train, y_pred)
    train_f2 = fbeta_score(y_train, y_pred, beta=2)
    
    metrics = {
        "precision": [train_prc, test_prc],
        "recall": [train_rcl, test_rcl],
        "f2score": [train_f2, test_f2],
    }

    return metrics