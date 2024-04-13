import os
import pandas as pd
import torch

from data_processor import DataProcessor

def main():
    # Get the data file path
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, 'data/Training_Essay_Data.csv')

    # Read the csv file
    data = pd.read_csv(data_path)

    data_processor = DataProcessor(data['text'].values)
    data_processor.process_data()

if __name__ == "__main__":
    main()
