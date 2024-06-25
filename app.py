from flask import Flask, session, jsonify, request
from diagnostics import model_predictions, dataframe_summary, execution_time, missing_values, outdated_packages_list
import json
import os
import subprocess


# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])

prediction_model = None


# Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    test_path = request.json.get('test_path')
    predictions = model_predictions(
        deployed_model_path=os.path.join(
            config['prod_deployment_path'],
            'trainedmodel.pkl'),
        test_data=test_path,
        response='exited'
    )
    return (jsonify(predictions))

# Scoring Endpoint


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def score():
    # check the score of the deployed model
    subprocess.run(["python", "scoring.py"])
    with open(os.path.join(config['output_model_path'], 'latestscore.txt')) as f:
        flat_list = [word for line in f for word in line.split()]
        score = float(flat_list[0])
    return (jsonify(score))  # add return value (a single F1 score number)

# Summary Statistics Endpoint


@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    # check means, medians, and modes for each column
    ss = dataframe_summary(
        data_path=os.path.join(config['output_folder_path'], 'finaldata.csv'),
        exclude=['exited'])
    return (jsonify(ss))  # return a list of all calculated summary statistics

# Diagnostics Endpoint


@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostics():
    # check timing and percent NA values
    timings = execution_time()
    missing = missing_values(
        data_path=os.path.join(
            config['output_folder_path'],
            'finaldata.csv'))
    dependencies = outdated_packages_list(requirements_path="requirements.txt")

    results = {
        'execution_time': timings,
        'missing_percentage': missing,
        'outdated_packages': dependencies
    }

    return (jsonify(results))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
