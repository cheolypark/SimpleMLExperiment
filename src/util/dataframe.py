import pandas as pd
import warnings
import datetime
import os

def get_os_path(file_path):
    return os.path.join(os.path.join(os.path.dirname(__file__)), file_path)

def load_file(file_path):
    return pd.read_csv(get_os_path(file_path))

def save_file(df, file_path, index=True):
    df.to_csv(get_os_path(file_path), index=index)
