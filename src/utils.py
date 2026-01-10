import os
import sys 
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill # to create pickle file
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import KFold

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        return CustomException(e,sys)
    
def kfold_cv(n_split):
    return KFold(n_splits =n_split)

def random_search_cv(model,param,cv,n_iter):
    return RandomizedSearchCV(estimator=model,param_distributions=param,cv=cv,n_iter=n_iter)

def grid_search_cv(estimator,param,cv):
    return GridSearchCV(estimator=estimator,param_grid=param,cv=cv)
 
def evaluate_model(X_train,y_train,X_test,y_test,models,params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]

            # use hypertuning method:
            clf = grid_search_cv(model,para,cv=kfold_cv(3))
            clf.fit(X_train,y_train)
            # model.fit(X_train,y_train)

            model.set_params(**clf.best_params_)

            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_pred)

            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path,mode='rb') as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        return CustomException(e,sys)




