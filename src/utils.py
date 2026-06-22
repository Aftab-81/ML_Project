import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score
from src.exception import CustomException

def save_object(file_path, object):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok = True)
        
        with open(file_path, "wb") as file:
            dill.dump(object, file)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params):

    try:
        report = {}
        best_models = {}

        for model_name in models.keys():

            model = models[model_name]
            para = params[model_name]

            gs = GridSearchCV(
                estimator=model,
                param_grid=para,
                cv=3,
                n_jobs=-1
            )

            gs.fit(X_train, y_train)

            model = gs.best_estimator_

            best_models[model_name] = model

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

            print(f"{model_name}")
            print(f"Best Parameters: {gs.best_params_}")
            print(f"Train R2 Score: {train_model_score:.4f}")
            print(f"Test R2 Score: {test_model_score:.4f}")
            print("-" * 50)

        return report, best_models

    except Exception as e:
        raise CustomException(e, sys)