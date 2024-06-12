import os
import json


# function for deployment
def deploy_model(model_path, score_path, ingested_data_path, deployment_path):
    """Copies the latest model, the latest score file, and the ingested data file into the deployment directory

    Args:
        model_path: path to trained model file
        score_path : path to latest score file
        ingested_data_path: path to ingested datasets list file
        deployment_dir: folder to save deployment files

    Return:
        None
    """

    # check if prod_deployment_path exists
    if not os.path.exists(deployment_path):
        os.makedirs(deployment_path)

    os.system(
        f"cp {model_path} {os.path.join(deployment_path, os.path.basename(model_path))}")
    os.system(
        f"cp {score_path} {os.path.join(deployment_path, os.path.basename(score_path))}")
    os.system(
        f"cp {ingested_data_path} {os.path.join(deployment_path, os.path.basename(ingested_data_path))}")


if __name__ == "__main__":

    # Load config.json and get path variables
    with open('config.json', 'r') as f:
        config = json.load(f)

    # deploy
    deploy_model(
        model_path=os.path.join(
            config['output_model_path'],
            'trainedmodel.pkl'),
        score_path=os.path.join(
            config['output_model_path'],
            'latestscore.txt'),
        ingested_data_path=os.path.join(
            config['output_folder_path'],
            'ingestedfiles.txt'),
        deployment_path=config['prod_deployment_path'])
