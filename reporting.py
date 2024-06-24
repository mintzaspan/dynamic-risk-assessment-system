# import pickle
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from diagnostics import model_predictions
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Function for reporting


def score_model(deployed_model_path, test_data, response, output_path):
    """Calculates a confusion matrix using the test data and the deployed model,
    then writes the confusion matrix to the workspace.
    """

    # calculate predictions first
    predictions = model_predictions(deployed_model_path, test_data, response)

    # confusion matrix
    y_test = pd.read_csv(test_data)[response]
    cm = confusion_matrix(y_true=y_test, y_pred=predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(os.path.join(output_path, 'confusionmatrix.png'))


if __name__ == '__main__':

    # Load config.json and get path variables
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Score
    score_model(deployed_model_path=os.path.join(
        config['prod_deployment_path'],
        'trainedmodel.pkl'),
        test_data=os.path.join(
            config['test_data_path'],
            'testdata.csv'),
        response='exited',
        output_path=config['output_model_path'])
