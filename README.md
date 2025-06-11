# Car Acceptability Prediction

**A Flask web app that predicts car acceptability using machine learning models (Random Forest, CatBoost, XGBoost). Deployed on Google Cloud Platform (GCP) App Engine standard environment using Google Cloud Shell, it offers a user-friendly interface for car evaluation predictions.**

# Repository Structure

```
├── templates/                  # HTML templates for Flask
├── app.py                     # Flask application
├── app.yaml                   # App Engine configuration
├── automotive-acceptance-modeling.ipynb  # Model training notebook
├── car.data                   # Dataset
├── cat_classifier.pkl         # CatBoost model
├── cat_metrics.pkl            # CatBoost metrics
├── label_encoders.pkl         # Label encoders
├── requirements.txt           # Python dependencies
├── rf_classifier.pkl          # Random Forest model
├── rf_metrics.pkl             # Random Forest metrics
├── xgb_metrics.pkl            # XGBoost metrics
├── xgboost_classifier.pkl     # XGBoost model
```

# Prerequisites

* Google Cloud Shell: Access via GCP Console (no local CLI needed).

* Python 3.9: App Engine standard runtime.

* GCP Project: With billing enabled.

* Git: Pre-installed in Cloud Shell.

# Setup and Deployment (Google Cloud Shell)

1. **Clone the Repository**

```
git clone https://github.com/jarif87/<repository-name>.git
cd <repository-name>
```

2. **Create a Google Cloud Project**
```
gcloud projects create car-prediction-<unique-id> --name="Car Prediction"
gcloud config set project car-prediction-<unique-id>
```

#### Enable APIs:
```
gcloud services enable appengine.googleapis.com cloudbuild.googleapis.com
```
3. **Initialize App Engine**
```
gcloud app init
```
4. **Update requirements.txt**
```
echo -e "flask==3.0.3\ngunicorn==22.0.0\nnumpy>=1.26.0\nscikit-learn==1.5.2\ncatboost==1.2.7\nxgboost==2.1.1\njoblib==1.4.2" > requirements.txt
```
5. **Configure app.yaml**
```
echo -e "runtime: python39\nentrypoint: gunicorn -b :\$PORT app:app\ninstance_class: F1" > app.yaml
```
6. **Deploy to App Engine**
```
gcloud app deploy app.yaml
```
#### View the app:
```
gcloud app browse
```
## Clean Up
```
gcloud projects delete car-prediction-<unique-id>
```





