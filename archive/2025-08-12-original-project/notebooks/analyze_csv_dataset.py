import pandas as pd
import numpy as np

def analyze_csv_dataset(file_path):
    """
    Analyzes a CSV dataset to determine its suitability for TDA.

    Args:
        file_path (str): The path to the CSV file.
    """
    print(f"Analyzing dataset: {file_path}")
    
    try:
        df = pd.read_csv(file_path)

        print("\n--- Dataset Info ---")
        df.info()

        print("\n--- Dataset Head ---")
        print(df.head())

        print("\n--- Summary Statistics ---")
        print(df.describe())

        print("\n--- Label Distribution ---")
        if 'label' in df.columns.str.lower():
            label_col = df.columns[df.columns.str.lower() == 'label'][0]
            print(df[label_col].value_counts())
        else:
            print("No 'Label' column found.")

    except Exception as e:
        print(f"Error analyzing dataset: {e}")

if __name__ == "__main__":
    # Update this path to the dataset you want to analyze
    dataset_path = "/home/stephen-dorman/dev/TDA_projects/data/apt_datasets/CIC-IDS-2017/Monday-WorkingHours.pcap_ISCX.csv"
    analyze_csv_dataset(dataset_path)
