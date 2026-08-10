


import pandas as pd
import numpy as np


def load_data(data_path):

    df = pd.read_csv(data_path, skipinitialspace = True, na_values = '?')
    print (f'Start data size: {df.shape} rows, {df.shape[1]} columns')

    initial_rows = len(df)
    df = df.dropna().reset_index(drop=True)
    print (f'Deleted missing values: {initial_rows - len(df)} rows deleted, {len(df)} rows remaining')

    target_col = 'income' if 'income' in df.columns else df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    classes = list(y.unique())
    print('Detected classes:', classes)
    return X,y,classes