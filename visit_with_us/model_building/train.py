
# for data manipulation
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# for model serialization
import joblib

# for creating a folder
import os

# for hugging face authentication
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# experiment tracking
import mlflow

mlflow.set_tracking_uri("https://mongrel-joylessly-hush.ngrok-free.dev")
mlflow.set_experiment("tourism-package-prediction")

api = HfApi()

# Load train/test datasets from HF Dataset Hub

Xtrain_path = "hf://datasets/ajeet04yadav/Tourism-Package-Prediction/Xtrain.csv"
Xtest_path = "hf://datasets/ajeet04yadav/Tourism-Package-Prediction/Xtest.csv"
ytrain_path = "hf://datasets/ajeet04yadav/Tourism-Package-Prediction/ytrain.csv"
ytest_path = "hf://datasets/ajeet04yadav/Tourism-Package-Prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# Convert to Series

ytrain = ytrain.squeeze()
ytest = ytest.squeeze()

print("Training and testing datasets loaded successfully.")

# Tourism Dataset Features

numeric_features = [
    'Age',
    'CityTier',
    'NumberOfPersonVisiting',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'Passport',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome',
    'PitchSatisfactionScore',
    'NumberOfFollowups',
    'DurationOfPitch'
]

categorical_features = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'MaritalStatus',
    'Designation',
    'ProductPitched'
]

# Handle class imbalance

class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Preprocessing pipeline

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# XGBoost Model

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42
)

# Hyperparameter Grid

param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__colsample_bytree': [0.6, 0.8, 1.0],
    'xgbclassifier__subsample': [0.8, 1.0]
}

# Pipeline

model_pipeline = make_pipeline(
    preprocessor,
    xgb_model
)

with mlflow.start_run():

    # Hyperparameter Tuning

    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        n_jobs=-1
    )

    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations

    results = grid_search.cv_results_

    for i in range(len(results['params'])):

        with mlflow.start_run(nested=True):

            mlflow.log_params(results['params'][i])

            mlflow.log_metric(
                "mean_test_score",
                results['mean_test_score'][i]
            )

            mlflow.log_metric(
                "std_test_score",
                results['std_test_score'][i]
            )

    # Best Parameters

    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    # Predictions

    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    train_report = classification_report(
        ytrain,
        y_pred_train,
        output_dict=True
    )

    test_report = classification_report(
        ytest,
        y_pred_test,
        output_dict=True
    )

    # Log Metrics

    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1_score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1_score": test_report['1']['f1-score']
    })

    # Save Model

    model_path = "best_tourism_package_model.joblib"

    joblib.dump(
        best_model,
        model_path
    )

    mlflow.log_artifact(
        model_path,
        artifact_path="model"
    )

    print(f"Model saved as artifact at: {model_path}")

    # Upload model to HF Model Hub

    repo_id = "ajeet04yadav/tourism-package-model"
    repo_type = "model"

    try:
        api.repo_info(
            repo_id=repo_id,
            repo_type=repo_type
        )

        print(f"Model repository '{repo_id}' already exists.")

    except RepositoryNotFoundError:

        create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=False
        )

        print(f"Model repository '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type
    )

    print("Model uploaded successfully.")
