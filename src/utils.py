import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException

def save_object(file_path, object):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok = True)
        
        with open(file_path, "wb") as file:
            dill.dump(object, file)
    except Exception as e:
        raise CustomException(e, sys)