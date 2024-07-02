import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


# Function for data ingestion
def merge_multiple_dataframes(input_path):
    """Checks for CSV files, compiles them together and writes to a CSV in the output path.
    Also lists ingested CSV files in txt in the output path.

    Args:
        input_path : path to check for CSV files

    Returns:
        csv_files : files used to creat final df
        final_df : final dataframe
    """

    csv_files = [f for f in os.listdir(input_path) if f.endswith('.csv')]

    df_list = []
    for file in csv_files:
        df = pd.read_csv(filepath_or_buffer=os.path.join(input_path, file))
        df_list.append(df)

    final_df = pd.concat(objs=df_list, axis=0, ignore_index=True)
    final_df.drop_duplicates(inplace=True, ignore_index=True)

    return (csv_files, final_df)


if __name__ == '__main__':

    # Load config.json and get input and output paths
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Ingest data
    csv_files, final_df = merge_multiple_dataframes(
        config['input_folder_path'])

    # create output dir if it does not exist
    if not os.path.exists(config['output_folder_path']):
        os.makedirs(config['output_folder_path'])

    # write df to CSV
    final_df.to_csv(
        path_or_buf=os.path.join(
            config['output_folder_path'],
            'finaldata.csv'),
        index=False)

    # write ingested filenames to txt
    with open(os.path.join(config['output_folder_path'], "ingestedfiles.txt"), 'w') as f:
        for file in csv_files:
            f.write(f"{os.path.join(config['input_folder_path'], file)}\n")
