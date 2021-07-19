import pandas as pd
import warnings
import datetime
import os

def load_file(file_path):
    os_path = os.path.join(os.path.join(os.path.dirname(__file__)), file_path)
    return pd.read_csv(os_path)

def save_file(df, file_path, index=True):
    os_path = os.path.join(os.path.join(os.path.dirname(__file__)), file_path)
    df.to_csv(os_path, index=index)
