from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import sys
import dill

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

app = application

# Route for home page
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods = ["GET", "POST"])
def pred_datapoint():
    if request.method == "GET":
        return render_template("home.html") # Go to input page
    else:
        try: 
            data = CustomData(
                gender = request.form.get("gender"),
                race_ethnicity = request.form.get("race_ethnicity"),
                parental_level_of_education = request.form.get("parental_level_of_education"),
                lunch = request.form.get("lunch"),
                test_preparation_course = request.form.get("test_preparation_course"),
                reading_score = request.form.get("reading_score"),
                writing_score = request.form.get("writing_score")
            )
            
            pred_input_df = data.get_data_as_data_frame() # Input in form of df

            print(pred_input_df)

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_input_df)

            return render_template("home.html", results = results[0])

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    app.run(host = "0.0.0.0")