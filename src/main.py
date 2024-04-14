import os
import pandas as pd

from data_processor import DataProcessor
from model import Model

def main():
    # Get the data file path
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, 'data/Training_Essay_Data.csv')

    # Read the csv file
    data = pd.read_csv(data_path)

    print("Setting up Model...")
    data_processor = DataProcessor(data['text'].values)

    model = Model(data_processor.matrix, data['generated'])
    print("Model complete.")

    while True:
        # Get user input
        user_input = input("Enter text: ")

        if user_input.lower() == "exit":
            break
        
        # Process the user input
        matrix_input = data_processor.get_input_matrix(user_input)
        prediction_prob = model.get_input_prediction(matrix_input)

        human_generated_prob = "{:.3f}".format(prediction_prob[0][0] * 100)
        ai_generated_prob = "{:.3f}".format(prediction_prob[0][1] * 100)

        print("\n", human_generated_prob, "% probability text is human generated.\n")
        print("\n", ai_generated_prob, "% probability text is AI generated.\n")

if __name__ == "__main__":
    main()
