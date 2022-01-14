import pandas as pd
import numpy as np
import math

def check_missing_data(df: pd.DataFrame) -> None:
    """
    Check the entire dataset for missing values
    """
    found = False
    for _, column in enumerate(df):
        if df[column].isnull().any():
            found = True
            print(f"{column} number on missing values is {df[column].isnull().sum()}")

    if not(found):
        print("There are no missing values in the dataset!")

def change_values(df, column, final_val, is_column_name = True, condition = lambda x: pd.isnull(x), transformations = lambda x: x):
    """
    Change the values in a given column based on some condition. New value can be given or derived from some other column.
    Transformations can be applied to the values from other columns.
    """
    for i, elem in enumerate(df[column]):
        if condition(elem):
            df.loc[i, column] = (transformations(df.loc[i, final_val]) if is_column_name else final_val)

shots = pd.read_csv('datasets/shot_logs.csv')

#preparing data for all numerical methods
shots['TIME_GAME_CLOCK'] = pd.to_datetime(shots['GAME_CLOCK'],format= '%M:%S' ).dt.minute * 60 + pd.to_datetime(shots['GAME_CLOCK'],format= '%M:%S' ).dt.second

shots['GAME_TIME'] = shots.apply(lambda row: (((row['PERIOD'] - 1) * 12 * 60 + (12 * 60 - row['TIME_GAME_CLOCK'])) if row['PERIOD'] <= 4 
                        else ((row['PERIOD'] - 5) * 5 * 60 + (5 * 60 - row['TIME_GAME_CLOCK'])) + 4 * 12 * 60), axis=1)

check_missing_data(shots)
change_values(shots, 'SHOT_CLOCK', 'TIME_GAME_CLOCK', transformations=lambda x: float(f"{math.floor(x/60)}.{x%60}"))
shots = shots[shots['TOUCH_TIME'] >= 0]

shots.to_csv("datasets/shot_logs_cleaned.csv")