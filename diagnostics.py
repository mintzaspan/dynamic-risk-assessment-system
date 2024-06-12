
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle


# Function to get model predictions
def model_predictions(deployed_model_path, test_data, response):
    """Reads the deployed model and a test dataset and returns calculated predictions in a list.

    Args:
        deployed_model_path: path to trained model marked for production
        test_data: path to data to predict on
        response: name of target variable

    Returns:
        predictions: list with predictions
    """

    # load model
    with open(deployed_model_path, 'rb') as f:
        model = pickle.load(f)

    # load data
    test_df = pd.read_csv(test_data)
    if response in test_df.columns:
        y_test = test_df[response]
        X_test = test_df.drop(columns=[response])
    else:
        X_test = test_df

    predictions = model.predict(X_test).tolist()
    print(predictions)

    return (predictions)

# Function to get summary statistics


def dataframe_summary():
    # calculate summary statistics here
    pass  # return value should be a list containing all summary statistics

# Function to get timings


def execution_time():
    # calculate timing of training.py and ingestion.py
    pass  # return a list of 2 timing values in seconds

# Function to check dependencies


def outdated_packages_list():
    pass  # get a list of


if __name__ == '__main__':

    # Load config.json and get environment variables
    with open('config.json', 'r') as f:
        config = json.load(f)

    # make predictions
    model_predictions(
        deployed_model_path=os.path.join(
            config['prod_deployment_path'],
            'trainedmodel.pkl'),
        test_data=os.path.join(
            config['test_data_path'],
            'testdata.csv'),
        response='exited')

    dataframe_summary()
    execution_time()
    outdated_packages_list()
