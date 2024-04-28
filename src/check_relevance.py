import pandas as pd
import scipy.stats as stats


def check_relevance(data):
    results = pd.DataFrame(columns=["Feature", "Correlation", "P_value"])

    # Assuming 'generated' is the name of your binary target variable
    target = data["generated"]

    for column in data.columns:
        if (
            column != "generated" and column != "text"
        ):  # Exclude the target variable and text columns
            try:
                # Ensure data is numeric for correlation computation
                numeric_data = pd.to_numeric(data[column], errors="coerce")
                # Drop NaN values that may result from conversion errors
                valid_data = numeric_data.dropna()
                valid_target = target[valid_data.index]

                # Calculate point-biserial correlation
                correlation, p_value = stats.pointbiserialr(valid_data, valid_target)

                # Create a DataFrame for the current results
                current_result = pd.DataFrame(
                    {
                        "Feature": [column],
                        "Correlation": [correlation],
                        "P_value": [p_value],
                    }
                )

                # Concatenate the current results to the main DataFrame
                results = pd.concat([results, current_result], ignore_index=True)
            except Exception as e:
                print(f"Could not process feature {column}: {e}")

    print(results)
