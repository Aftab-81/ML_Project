import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException

def save_object(file_path, object):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok = True)
        
        with open(file_path, "wb") as file:
            dill.dump(object, file)
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models):
    report = {}
    try:
        for i in range(1, len(models)):
            model = list(models.keys())[i]

            model = models[model]
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
            
        return report
    except Exception as e:
        raise CustomException(e, sys)