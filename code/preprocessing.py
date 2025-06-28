import os
import pandas as pd


def load_data():
    raw_data_path = os.path.join('..', 'data', 'raw', 'UNSW_NB15')

    train_file = os.path.join(raw_data_path, 'UNSW_NB15_training-set.csv')
    test_file = os.path.join(raw_data_path, 'UNSW_NB15_testing-set.csv')

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    df = pd.concat([train_df, test_df], ignore_index=True)

    return df
