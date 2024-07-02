import pandas as pd
import pickle
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import json


# Function for training the model
def train_model(data_path, response):
    """Trains a Logistic Regression model

    Args:
        data_path : path of CSV dataset
        response: response variable
        model_output_path: folder to save trained model as trainedmodel.pkl

    Returns:
        pipe: a pipeline including a preprocessing step and a trained Logistic Regression model
    """

    data = pd.read_csv(data_path)
    y = data[response]
    X = data.drop(columns=[response])

    # preprocessing
    num_features = X.select_dtypes(include='number').columns.tolist()
    num_transformer = make_pipeline(SimpleImputer(), StandardScaler())

    cat_features = X.select_dtypes(include='object').columns.tolist()
    cat_transformer = make_pipeline(
        SimpleImputer(
            strategy='most_frequent'), TargetEncoder(
            target_type='binary'), StandardScaler())

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
        ]
    )

    # use this logistic regression for training
    logreg = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class='deprecated',
        n_jobs=None,
        penalty='l2',
        random_state=0,
        solver='liblinear',
        tol=0.0001,
        verbose=0,
        warm_start=False)

    # model pipeline
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor), ("classifier", logreg)])

    # fit the logistic regression to your data
    pipe.fit(X, y)

    return (pipe)


if __name__ == "__main__":

    # Load config.json and get path variables
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Train model and save as pkl
    pipe = train_model(
        data_path=os.path.join(config['output_folder_path'], 'finaldata.csv'),
        response="exited")

    # check if model_output_path exists
    if not os.path.exists(config['output_model_path']):
        os.makedirs(config['output_model_path'])

    # write the trained model to your workspace in a file called
    # trainedmodel.pkl
    with open(os.path.join(config['output_model_path'], "trainedmodel.pkl"), "wb") as f:
        pickle.dump(obj=pipe, file=f)
