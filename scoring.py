import pandas as pd
import pickle
import os
from sklearn.metrics import fbeta_score
import json


# Function for model scoring
def score_model(model_path, test_data_path, response):
    """Takes a trained model, loads test data, and calculates an F1 score.

    Args:
        model_path: path to trained model
        test_data_path: path to test data
        response: response variable

    Returns:
        f1_score : F1 score
    """

    # read test data
    test_df = pd.read_csv(test_data_path)
    y_test = test_df[response]
    X_test = test_df.drop(columns=[response])

    # load trained model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # calculate F1 score
    y_pred = model.predict(X_test)
    f1_score = fbeta_score(y_true=y_test, y_pred=y_pred, beta=1)

    return (f1_score)


if __name__ == '__main__':

    # Load config.json and get path variables
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Score model
    f1_score = score_model(
        model_path=os.path.join(
            config['output_model_path'],
            'trainedmodel.pkl'),
        test_data_path=os.path.join(
            config['test_data_path'],
            'testdata.csv'),
        response="exited")

    # save to file
    if not os.path.exists(config['output_model_path']):
        os.makedirs(config['output_model_path'])

    with open(os.path.join(config['output_model_path'], "latestscore.txt"), 'w') as f:
        f.write(f"{f1_score}")
