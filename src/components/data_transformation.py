import sys
import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from dataclasses import dataclass
from src.utils import save_object
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    # final_models\preprocessor.pkl
    preprocessor_obj_file_path = os.path.join("final_models", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation
        """
        try:
            num_features = ["reading score", "writing score"]
            cat_features = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]
            
            num_pipeline = Pipeline(
                steps=[
                    ("Simple Imputer", SimpleImputer(strategy = 'median')),
                    ("Standard Scaler", StandardScaler())
                ]
            )

            logging.info("Numerical columns Standard Scaling completed")


            cat_pipeline = Pipeline(
                steps = [
                    ("Simple Imputer", SimpleImputer(strategy = 'most_frequent')),
                    ("One Hot Encoder", OneHotEncoder()),
                    ("Standard Scaler", StandardScaler(with_mean = False))
                ]
            )

            """
            Error For: ("Standard Scaler", StandardScaler()) 
            [Cannot center sparse matrices: pass `with_mean=False` instead. See docstring for motivation and alternatives.]

            Reason:

            -- Standard Scaler does => (x - mean) this operation called as centering
            -- One Hot Encoding create sparse matrix. 
            -- Sparse matrices don't support centering efficiently, so Scikit-Learn throws:
                    "Cannot center sparse matrices"

            Solution:
                        Use =>   with_mean=False    (Don't subtract the mean from the data)
                    
            """

            logging.info("Categorical Columns Encoding Completed")

            preprocessor = ColumnTransformer(
                transformers = [
                    ("Numerical Pipeline", num_pipeline, num_features),
                    ("Categorical Pipeline", cat_pipeline, cat_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):  #  (train_path, test_path) => from Data Ingestion
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read the train and test data")

            logging.info("Obtaining preprocessor object")

            preprocessor = self.get_data_transformer_object()

            logging.info("Obtained the preprocessor object")

            target_column_name = "math score"
            num_features = ["reading score", "writing score"]

            # X_train
            input_feature_train_df = train_df.drop(target_column_name, axis = 1)
            # y_train
            target_feature_train_df = train_df[target_column_name]

            # X_test
            input_feature_test_df = test_df.drop(target_column_name, axis = 1)
            
            # test_df[[target_column_name]] if we do this, then no need to write np.array() while creating train_arr and test_arr

            # y_test
            target_feature_test_df = test_df[target_column_name] 

            logging.info("Applying preprocessing object on training and testing dataframes")

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            """
            we did this because we transformed X and y and now we want give it to model 
            but giving data seperately to model is not possible so we combine it into a dataframe format.
            """

            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                object = preprocessor
            )

            logging.info("Saved preprocessor object")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
