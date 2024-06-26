import json
import os
import glob
import sys
import subprocess


# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)


# # Check and read new data
# # first, read ingestedfiles.txt
# with open(os.path.join(config['output_folder_path'], 'ingestedfiles.txt'), 'r') as file:
#     ingested_files = [line.replace('\n', '') for line in file.readlines()]

# # second, determine whether the source data folder has files that aren't
# # listed in ingestedfiles.txt
# new_files = []
# for file in glob.glob(os.path.join(config['input_folder_path'], '*.csv')):
#     if file not in ingested_files:
#         new_files.append(file)


# # Deciding whether to proceed, part 1
# # if you found new data, you should proceed. otherwise, do end the process here
# if len(new_files) == 0:
#     sys.exit(0)
# else:
#     subprocess.run(["python", "ingestion.py"])

# Checking for model drift
# check whether the score from the deployed model is different from the
# score from the model that uses the newest ingested data]
with open(os.path.join(config['prod_deployment_path'], 'latestscore.txt'), 'r') as file:
    current_score = float(file.read().strip())

# get new score
subprocess.run(["python", "training.py"])
subprocess.run(["python", "scoring.py"])

with open(os.path.join(config['output_model_path'], 'latestscore.txt'), 'r') as file:
    new_score = float(file.read().strip())

# Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the
# process here
if new_score <= current_score:
    sys.exit(0)
# Re-deployment
# if you found evidence for model drift, re-run the deployment.py script
else:
    subprocess.run(["python", "deployment.py"])
    # Diagnostics and reporting
    # run diagnostics.py and reporting.py for the re-deployed model
    subprocess.run(["python", "diagnostics.py"])
    subprocess.run(["python", "reporting.py"])
    subprocess.run(["python", "app.py"])
    subprocess.run(["python", "apicalls.py"])
