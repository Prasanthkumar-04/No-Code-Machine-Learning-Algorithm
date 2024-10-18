import pandas as pd

def load_data(file_path):

    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def preprocess_data(df, target_column):

    return df
