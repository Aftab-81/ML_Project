import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_model

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    model_obj_file_path = os.path.join("final_models", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr, preprocessor_path):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1], 
                train_arr[:, -1], 
                test_arr[:, :-1], 
                test_arr[:, -1]
            )
            logging.info("Splitted the data into train and test")


            models = {
                "Linear Regression": LinearRegression(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose = False),
                "AdaBoostRegressor": AdaBoostRegressor()
            }

            params = {

    "Linear Regression": {
        "fit_intercept": [True, False]
    },

    "Gradient Boosting Regressor": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5]
    },

    "KNeighborsRegressor": {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"]
    },

    "DecisionTreeRegressor": {
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5]
    },

    "RandomForestRegressor": {
        "n_estimators": [100, 200],
        "max_depth": [10, 20],
        "min_samples_split": [2, 5]
    },

    "XGBRegressor": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5]
    },

    "CatBoostRegressor": {
        "iterations": [100, 200],
        "learning_rate": [0.05, 0.1],
        "depth": [4, 6]
    },

    "AdaBoostRegressor": {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1]
    }
}

            logging.info("Requesting for model report")
            model_report, best_models = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )
            logging.info("Model report is received")

            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[ # get index value
                list(model_report.values()).index(best_model_score) # index which having best_model_score
            ]
                
            best_model = best_models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("Best model found for both train and test data")


            save_object(
                file_path = self.model_trainer_config.model_obj_file_path,
                object = best_model
                )
            
            logging.info("Best model is saved")
            

            y_test_pred = best_model.predict(X_test)

            return r2_score(y_test, y_test_pred)

        except Exception as e:
            raise CustomException(e, sys)