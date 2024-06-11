import os
import json


# function for deployment
def deploy_model(model_path, ingested_data_path, deployment_path):
    """Copys the latest pickle file, the latestscore.txt value, and the ingestedfiles.txt file into the deployment directory

    Args:
        model_path: folder that contains trainedmodel.pkl and latestscore.txt
        ingested_data_path: folder that contains ingestedfiles.txt

    Return:
        None
    """

    # check if prod_deployment_path exists
    if not os.path.exists(deployment_path):
        os.makedirs(deployment_path)

    os.system(
        f"cp {os.path.join(model_path, 'trainedmodel.pkl')} {os.path.join(deployment_path, 'trainedmodel.pkl')}")
    os.system(
        f"cp {os.path.join(model_path, 'latestscore.txt')} {os.path.join(deployment_path, 'latestscore.txt')}")
    os.system(
        f"cp {os.path.join(ingested_data_path, 'ingestedfiles.txt')} {os.path.join(deployment_path, 'ingestedfiles.txt')}")


if __name__ == "__main__":

    # Load config.json and get path variables
    with open('config.json', 'r') as f:
        config = json.load(f)

    dataset_csv_path = os.path.join(config['output_folder_path'])
    model_path = os.path.join(config['output_model_path'])
    prod_deployment_path = os.path.join(config['prod_deployment_path'])

    # deploy
    deploy_model(
        model_path=model_path,
        ingested_data_path=dataset_csv_path,
        deployment_path=prod_deployment_path)
