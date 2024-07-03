import json
import os
import glob
import sys
import subprocess
from ingestion import merge_multiple_dataframes
from training import train_model
from scoring import score_model
import pickle


# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)


# Check and read new data
# First, read ingestedfiles.txt
with open(os.path.join(config['output_folder_path'], 'ingestedfiles.txt'), 'r') as file:
    ingested_files = [line.replace('\n', '') for line in file.readlines()]

# Second, determine whether the source data folder has files that aren't
# listed in ingestedfiles.txt
new_files = []
for file in glob.glob(os.path.join(config['input_folder_path'], '*.csv')):
    if file not in ingested_files:
        new_files.append(file)


# Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here
if len(new_files) == 0:
    decision_1 = False
else:
    decision_1 = True

# Checking for model drift
# check whether the score from the deployed model is different from the
# score from the model that uses the newest ingested data]
with open(os.path.join(config['prod_deployment_path'], 'latestscore.txt'), 'r') as file:
    current_score = float(file.read().strip())

# # # get new score
if not os.path.exists('check/'):
    os.makedirs('check/')
new_files, new_data = merge_multiple_dataframes(
    input_path=config['input_folder_path'])
new_data.to_csv('check/new_data.csv', index=False)
new_model = train_model('check/new_data.csv', response='exited')
with open("check/trainedmodel.pkl", "wb") as f:
    pickle.dump(obj=new_model, file=f)
new_score = score_model('check/trainedmodel.pkl', os.path.join(
    config['test_data_path'],
    'testdata.csv'),
    'exited')
subprocess.run(['rm', '-rf', 'check/'])

# Proceed or not, part two
if new_score > current_score:
    decision_2 = True
else:
    decision_2 = False

if (decision_1 & decision_2):
    subprocess.run(['python', 'ingestion.py'])
    subprocess.run(['python', 'training.py'])
    subprocess.run(['python', 'scoring.py'])
    subprocess.run(['python', 'deployment.py'])
    subprocess.run(['python', 'reporting.py'])
    subprocess.run(['python', 'apicalls.py'])
