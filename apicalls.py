import requests
import os
import json

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"

# read config
with open('config.json', 'r') as file:
    config = json.load(file)
    model_path = os.path.join(config['output_model_path'])


# Call each API endpoint and store the responses
response1 = requests.post(
    url=os.path.join(URL, 'prediction'),
    json={'test_path': os.path.join(config['test_data_path'], 'testdata.csv')}
).text

response2 = requests.get(os.path.join(URL, 'scoring')).text
response3 = requests.get(os.path.join(URL, 'summarystats')).text
response4 = requests.get(os.path.join(URL, 'diagnostics')).text

# combine all API responses
responses = {"prediction": response1,
             "scoring": response2,
             "summarystats": response3,
             "diagnostics": response4
             }

# #write the responses to your workspace
with open(os.path.join(config['output_model_path'], "apireturns.txt"), "w") as file:
    file.write(json.dumps(responses))
