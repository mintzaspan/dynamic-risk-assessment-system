import pandas as pd
import pickle
import os
from sklearn.metrics import fbeta_score
import json


# Function for model scoring
def score_model(model_path, test_data_path, response):
    """Takes a trained model, loads test data, and calculates an F1 score for the model relative to the test data.
    Then writes the result to the latestscore.txt file

    Args:
        model_path: path to trained model
        test_data_path: path to test data
        response: response variable

    Returns:
        None
    """

    # read test data
    test_df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    y_test = test_df[response]
    X_test = test_df.drop(columns=[response])

    # load trained model
    with open(os.path.join(model_path, "trainedmodel.pkl"), "rb") as f:
        model = pickle.load(f)

    # calculate F1 score
    y_pred = model.predict(X_test)
    f1_score = fbeta_score(y_true=y_test, y_pred=y_pred, beta=1)

    # save to file
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    with open(os.path.join(model_path, "latestscore.txt "), 'w') as f:
        f.write(f"{f1_score}\n")


if __name__ == '__main__':

    # Load config.json and get path variables
    with open('config.json', 'r') as f:
        config = json.load(f)

    model_path = os.path.join(config['output_model_path'])
    test_data_path = os.path.join(config['test_data_path'])

    # Score model
    score_model(
        model_path=model_path,
        test_data_path=test_data_path,
        response="exited")
