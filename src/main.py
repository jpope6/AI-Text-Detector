import os
import pandas as pd
import numpy as np
from data_processor import DataProcessor
from add_features import add_features
from model import Model
from scipy.sparse import hstack, csr_matrix
from check_relevance import check_relevance


def main():
    # Determine the current directory and construct the path to the data file.
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, "data/Training_Essay_Data.csv")

    # Load the dataset from the specified CSV file.
    data = pd.read_csv(data_path)

    # Notify the user that model setup is beginning.
    print("Setting up Model...")

    # Apply feature engineering to add additional features to the dataset.
    data_with_more_features = add_features(data)

    # Check how relivant each feature is to predicting "generated"
    # check_relevance(data_with_more_features)

    # Initialize the data processor with the enhanced data, perform preprocessing and feature extraction.
    data_processor = DataProcessor(data_with_more_features)

    # Initialize the model with the reduced feature set from the data processor and the target variable.
    model = Model(data_processor.combined_features_reduced, data["generated"])

    # Confirm that the model setup is complete.
    print("Model complete.")

    # Enter an infinite loop to continuously accept user input.
    while True:
        # Prompt the user for text input.
        user_input = input("Enter text: ")

        # Exit the loop if the user types 'exit'.
        if user_input.lower() == "exit":
            break

        # Convert user input into a DataFrame to maintain consistency with data processing.
        user_input_df = pd.DataFrame({"text": [user_input]})

        # Apply the same feature engineering process to the user input.
        user_input_with_more_features = add_features(user_input_df)

        # Convert the processed user input text into a numerical matrix using the data processor.
        matrix_input = data_processor.get_input_matrix(user_input)

        # Extract other features from the user input needed for the model.
        other_features = user_input_with_more_features[
            [
                "char_count",
                "word_count",
                "capital_char_count",
                "capital_word_count",
                "punctuation_count",
                "quoted_word_count",
                "sent_count",
                "unique_word_count",
                "stopword_count",
                "avg_word_length",
                "avg_sent_length",
                "unique_vs_words",
                "stopwords_vs_words",
            ]
        ]

        # Convert the extracted features into a sparse matrix format.
        other_features_sparse = csr_matrix(other_features.values)

        # Combine the TF-IDF matrix and other features into one feature set.
        input_features_combined = hstack((matrix_input, other_features_sparse))

        # Reduce the dimensionality of the combined features using pre-trained SVD.
        input_features_reduced = data_processor.svd.transform(input_features_combined)

        # Make a prediction using the model based on the processed features.
        prediction_prob = model.get_input_prediction(input_features_reduced)

        # Format and display the probabilities of the text being human or AI-generated.
        human_generated_prob = "{:.3f}".format(prediction_prob[0][0] * 100)
        ai_generated_prob = "{:.3f}".format(prediction_prob[0][1] * 100)

        print("\n", human_generated_prob, "% probability text is human generated.\n")
        print("\n", ai_generated_prob, "% probability text is AI generated.\n")


if __name__ == "__main__":
    main()
