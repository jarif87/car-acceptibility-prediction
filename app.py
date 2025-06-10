from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
import os
from werkzeug.exceptions import BadRequestKeyError

app = Flask(__name__)

# Define file paths
cat_model = "cat_classifier.pkl"
cat_metrics = "cat_metrics.pkl"
rf_model = "rf_classifier.pkl"
rf_metrics = "rf_metrics.pkl"
xgb_model = "xgboost_classifier.pkl"
xgb_metrics = "xgb_metrics.pkl"
label_encoder = "label_encoders.pkl"

# Load models, metrics, and encoders
try:
    with open(cat_model, 'rb') as file:
        cat_classifier = pickle.load(file)
    with open(cat_metrics, 'rb') as file:
        cat_metrics_dict = pickle.load(file)
    with open(rf_model, 'rb') as file:
        rf_classifier = pickle.load(file)
    with open(rf_metrics, 'rb') as file:
        rf_metrics_dict = pickle.load(file)
    with open(xgb_model, 'rb') as file:
        xgb_classifier = pickle.load(file)
    with open(xgb_metrics, 'rb') as file:
        xgb_metrics_dict = pickle.load(file)
    with open(label_encoder, 'rb') as file:
        label_encoders = pickle.load(file)
except FileNotFoundError as e:
    print(f"Error loading file: {e}")
    raise

# Define algorithm mapper
algomapper = {
    'rf': (rf_classifier, rf_metrics_dict),
    'cat': (cat_classifier, cat_metrics_dict),
    'xgb': (xgb_classifier, xgb_metrics_dict)
}

# Define class mappings
classmapper = {0: 'Acceptable', 1: 'Good', 2: 'Unacceptable', 3: 'Very Good'}

# Define feature mappings for EDA display
values = ["Buying Price", "Maintenance Cost", "Number of Doors", "Passenger Capacity", "Luggage Boot Size", "Safety Level"]
keys = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
mapper = dict(zip(keys, values))

# Read data for EDA
def readdata():
    colnames = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    try:
        df = pd.read_csv("car.data", names=colnames)
        return df
    except FileNotFoundError:
        raise Exception("car.data file not found")

# Groupby for EDA
def getdict(colname):
    df = readdata()
    return dict(df.groupby([colname])['class'].count())

@app.route("/", methods=["GET", "POST"])
def hello_world():
    error_message = None
    if request.method == "POST":
        try:
            mydict = request.form
            # Validate form inputs
            required_fields = ['buy', 'maintain', 'doors', 'person', 'luggage', 'safety', 'algo', 'value']
            for field in required_fields:
                if field not in mydict:
                    raise BadRequestKeyError(field)

            # Convert form inputs to integers
            buy = int(mydict['buy'])
            maintain = int(mydict['maintain'])
            doors = int(mydict['doors'])
            person = int(mydict['person'])
            luggage = int(mydict['luggage'])
            safety = int(mydict['safety'])
            algo = mydict['algo']
            value = mydict['value']

            # Validate algorithm
            if algo not in algomapper:
                raise ValueError("Invalid algorithm selected")

            # Get value counts for EDA
            valuecount = getdict(value)

            # Select model and metrics
            model, metrics_dict = algomapper[algo]
            accuracy = metrics_dict['accuracy']
            precision = metrics_dict['precision']
            recall = metrics_dict['recall']
            f1score = metrics_dict['f1score']

            # Prepare input for prediction
            inputparam = np.array([[buy, maintain, doors, person, luggage, safety]])

            # Make prediction
            predict = model.predict(inputparam)
            predictedclass = classmapper[predict[0]]

            # Model comparison data
            model_comparison = [
                {'name': 'Random Forest', 'accuracy': round(rf_metrics_dict['accuracy']*100, 2), 'precision': round(rf_metrics_dict['precision'], 2)},
                {'name': 'CatBoost', 'accuracy': round(cat_metrics_dict['accuracy']*100, 2), 'precision': round(cat_metrics_dict['precision'], 2)},
                {'name': 'XGBoost', 'accuracy': round(xgb_metrics_dict['accuracy']*100, 2), 'precision': round(xgb_metrics_dict['precision'], 2)}
            ]

            return render_template('index.html',
                                   predictedclass=predictedclass,
                                   display=True,
                                   accuracy=round(accuracy*100, 2),
                                   precision=round(precision, 2),
                                   recall=round(recall, 2),
                                   f1score=round(f1score, 2),
                                   showtemplate=True,
                                   valuecount=valuecount,
                                   value=mapper[value],
                                   mapper=valuecount,
                                   model_comparison=model_comparison,
                                   error_message=None)

        except (BadRequestKeyError, ValueError, KeyError, Exception) as e:
            error_message = f"Invalid input or error: {str(e)}. Please ensure all fields are filled correctly and car.data is available."

    return render_template('index.html', error_message=error_message, model_comparison=[
        {'name': 'Random Forest', 'accuracy': round(rf_metrics_dict['accuracy']*100, 2), 'precision': round(rf_metrics_dict['precision'], 2)},
        {'name': 'CatBoost', 'accuracy': round(cat_metrics_dict['accuracy']*100, 2), 'precision': round(cat_metrics_dict['precision'], 2)},
        {'name': 'XGBoost', 'accuracy': round(xgb_metrics_dict['accuracy']*100, 2), 'precision': round(xgb_metrics_dict['precision'], 2)}
    ])

if __name__ == '__main__':
    port=int(os.environ.get("PORT",8080))
    app.run(debug=True,host="0.0.0.0",port=port)