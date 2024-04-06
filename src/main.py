import os
import pandas as pd

def main():
    # Get the data file path
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, 'data/Training_Essay_Data.csv')

    # Read the csv file
    data = pd.read_csv(data_path)
    print(data.head(15))

if __name__ == "__main__":
    main()
