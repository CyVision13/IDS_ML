import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def load_data():
    raw_data_path = os.path.join('..', 'data', 'raw', 'UNSW_NB15')

    train_file = os.path.join(raw_data_path, 'UNSW_NB15_training-set.csv')
    test_file = os.path.join(raw_data_path, 'UNSW_NB15_testing-set.csv')

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    df = pd.concat([train_df, test_df], ignore_index=True)

    return df


def clean_data(df):
    print("Missing values before cleaning:")
    print(df.isnull().sum())

    df = df.dropna()

    df = df.drop_duplicates()

    print("Missing values after cleaning:")
    print(df.isnull().sum())

    print(f"Number of duplicates removed: {len(df) - len(df.drop_duplicates())}")

    return df


def normalize_numerical_features(df, numerical_cols):
    """Apply min-max normalization to numerical features"""
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df


def encode_categorical_features(df, categorical_cols):
    """Convert categorical features to numerical using label encoding"""
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le  # Save encoders if needed later

    return df, encoders


def preprocess_data():
    df = load_data()

    df = clean_data(df)

    categorical_cols = ['proto', 'service', 'state', 'attack_cat']
    numerical_cols = [col for col in df.columns if col not in categorical_cols + ['label', 'id']]

    df = normalize_numerical_features(df, numerical_cols)

    df, encoders = encode_categorical_features(df, categorical_cols)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    # Save processed data
    processed_path = os.path.join('..', 'data', 'processed')
    os.makedirs(processed_path, exist_ok=True)

    train_df.to_csv(os.path.join(processed_path, 'train_processed.csv'), index=False)
    test_df.to_csv(os.path.join(processed_path, 'test_processed.csv'), index=False)

    print("\nPreprocessing complete!")
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")

    return train_df, test_df


if __name__ == '__main__':
    train_df, test_df = preprocess_data()
