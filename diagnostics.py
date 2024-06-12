
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

    return (predictions)

# Function to get summary statistics


def dataframe_summary(data_path, exclude):
    """Calculates summary statistics for numericals columns of a dataset

    Args:
        data_path: path to CSV data file
        exclude: list of columns to exclude

    Returns:
        stats: list of lists with column name, mean, median, standard deviation
    """

    # import data
    df = pd.read_csv(data_path).select_dtypes('number')
    df.drop(columns=exclude, inplace=True, errors='ignore')

    # calculate statistics
    stats = []
    for i in df.columns.tolist():
        mean = df[i].mean()
        median = df[i].median()
        std = df[i].std()
        stats.append([i, mean, median, std])

    return (stats)

# Function to get timings


def execution_time():
    """Calculate execution timings of training.py and ingestion.py

    Args:
        None

    Returns:
        timings: list of execution timings of training.py and ingestion.py
    """

    ingestion_starttime = timeit.default_timer()
    os.system('python ingestion.py')
    ingestion_timing = timeit.default_timer() - ingestion_starttime

    training_starttime = timeit.default_timer()
    os.system('python training.py')
    training_timing = timeit.default_timer() - training_starttime

    return ([ingestion_timing, training_timing])

# Function to check for missing values


def missing_values(data_path):
    """Calculates the percentage of missing values for every column in a given dataset.

    Args:
        data path: path to CSV data file

    Returns:
        missing_stats: list of missing values percentage by column
    """
    # import data
    df = pd.read_csv(data_path)

    missing = (pd.isna(df).sum() / len(df)).to_list()

    return (missing)


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

    dataframe_summary(
        data_path=os.path.join(config['output_folder_path'], 'finaldata.csv'),
        exclude=['exited']
    )

    # time ingestion and training
    execution_time()

    # check for missing values
    missing_values(
        data_path=os.path.join(
            config['output_folder_path'],
            'finaldata.csv'))

    outdated_packages_list()
