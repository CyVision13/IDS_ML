
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def load_data():
    """Load NSL-KDD dataset from raw data files"""
    # Get the absolute path to ensure correct file loading
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(current_dir, '..', 'data', 'raw', 'NSL_KDD')
    train_file = os.path.join(raw_data_path, 'KDDTrain+.TXT')
    test_file = os.path.join(raw_data_path, 'KDDTest+.TXT')

    # Check if files exist
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    column_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
        'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
    ]

    try:
        print("Loading training data...")
        train_df = pd.read_csv(train_file, names=column_names)
        print(f"Training data loaded: {len(train_df)} records")
        
        print("Loading test data...")
        test_df = pd.read_csv(test_file, names=column_names)
        print(f"Test data loaded: {len(test_df)} records")
        
        df = pd.concat([train_df, test_df], ignore_index=True)
        print(f"Combined dataset: {len(df)} records")
        
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def load_data_separately():
    """Load NSL-KDD train and test datasets separately as per article methodology"""
    # Get the absolute path to ensure correct file loading
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(current_dir, '..', 'data', 'raw', 'NSL_KDD')
    train_file = os.path.join(raw_data_path, 'KDDTrain+.TXT')
    test_file = os.path.join(raw_data_path, 'KDDTest+.TXT')

    # Check if files exist
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    column_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
        'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
    ]

    try:
        print("Loading training data...")
        train_df = pd.read_csv(train_file, names=column_names)
        print(f"Training data loaded: {len(train_df)} records")
        
        print("Loading test data...")
        test_df = pd.read_csv(test_file, names=column_names)
        print(f"Test data loaded: {len(test_df)} records")
        
        return train_df, test_df
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def clean_data(df):
    """Clean the dataset by removing missing values and duplicates"""
    print("\n=== Data Cleaning ===")
    print(f"Initial dataset shape: {df.shape}")
    
    print("Missing values before cleaning:")
    missing_counts = df.isnull().sum()
    print(missing_counts[missing_counts > 0])
    
    if missing_counts.sum() == 0:
        print("No missing values found.")
    
    # Remove missing values
    initial_len = len(df)
    df = df.dropna()
    print(f"Records removed due to missing values: {initial_len - len(df)}")
    
    # Remove duplicates
    initial_len = len(df)
    df = df.drop_duplicates()
    print(f"Duplicate records removed: {initial_len - len(df)}")
    
    print(f"Final dataset shape after cleaning: {df.shape}")
    
    return df


def normalize_numerical_features(df, numerical_cols):
    """Apply min-max normalization to numerical features"""
    print(f"\nNormalizing {len(numerical_cols)} numerical features...")
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    print("Normalization completed.")
    return df, scaler


def preprocess_dataset(df, dataset_name, scaler=None, fit_scaler=True):
    """Preprocess a single dataset (train or test)"""
    print(f"\n=== Preprocessing {dataset_name} Dataset ===")
    
    # Clean data
    df = clean_data(df)
    
    # Display label distribution before processing
    print(f"\nLabel distribution in {dataset_name} before processing:")
    print(df['label'].value_counts())
    
    # Drop 'difficulty' column (not used in modeling)
    if 'difficulty' in df.columns:
        df = df.drop(columns=['difficulty'])
        print("Dropped 'difficulty' column")
    
    return df


def apply_label_encoding(df, binary_label=False):
    """Apply label encoding to dataset"""
    if not binary_label:
        print("Converting labels to 5-class format as per article...")
        
        # Attack mapping from article - maps specific attacks to main categories
        attack_mapping = {
            # DoS attacks
            'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 'smurf': 'DoS', 
            'teardrop': 'DoS', 'apache2': 'DoS', 'udpstorm': 'DoS', 'processtable': 'DoS', 
            'worm': 'DoS',
            # Probe attacks
            'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 
            'mscan': 'Probe', 'saint': 'Probe',
            # R2L attacks
            'guess_passwd': 'R2L', 'ftp_write': 'R2L', 'imap': 'R2L', 'phf': 'R2L', 
            'multihop': 'R2L', 'warezmaster': 'R2L', 'warezclient': 'R2L', 'spy': 'R2L', 
            'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguess': 'R2L', 'snmpgetattack': 'R2L', 
            'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
            # U2R attacks
            'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R', 'perl': 'U2R', 
            'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R'
        }
        
        # Map specific attacks to main categories, keep 'normal' as is
        df['label'] = df['label'].apply(lambda x: attack_mapping.get(x, 'normal'))
        
        # Final numerical mapping for 5 classes
        final_label_mapping = {
            'normal': 0,
            'DoS': 1,
            'Probe': 2,
            'R2L': 3,
            'U2R': 4
        }
        
        df['label'] = df['label'].map(final_label_mapping)
        
    else:
        # Binary classification
        print("Converting labels to binary format...")
        df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    
    print("Label distribution after conversion:")
    print(df['label'].value_counts())
    
    return df


def apply_feature_encoding(df, categorical_cols=None):
    """Apply one-hot encoding to categorical features"""
    if categorical_cols is None:
        categorical_cols = ['protocol_type', 'service', 'flag']
    
    # Verify categorical columns exist
    existing_categorical_cols = [col for col in categorical_cols if col in df.columns]
    if len(existing_categorical_cols) != len(categorical_cols):
        missing_cols = set(categorical_cols) - set(existing_categorical_cols)
        print(f"Warning: Missing categorical columns: {missing_cols}")
    
    print(f"Applying one-hot encoding to categorical features: {existing_categorical_cols}")
    
    # Separate label before encoding
    labels = df['label']
    features_df = df.drop('label', axis=1)
    
    # Apply one-hot encoding
    features_encoded = pd.get_dummies(features_df, columns=existing_categorical_cols, dummy_na=False)
    
    # Add label back
    df_encoded = features_encoded.copy()
    df_encoded['label'] = labels
    
    print(f"Features after one-hot encoding: {len(df_encoded.columns)-1}")  # -1 for label
    
    return df_encoded


def apply_normalization(df, scaler=None, fit_scaler=True):
    """Apply min-max normalization to numerical features"""
    # Identify numerical columns for normalization (exclude binary and label columns)
    binary_cols = ['land', 'logged_in', 'is_host_login', 'is_guest_login', 'label']
    # Get one-hot encoded categorical columns
    categorical_prefixes = ['protocol_type', 'service', 'flag']
    encoded_categorical_cols = [col for col in df.columns if any(cat in col for cat in categorical_prefixes)]
    
    exclude_cols = binary_cols + encoded_categorical_cols
    numerical_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"\nFeature categorization:")
    print(f"  One-hot encoded categorical features: {len(encoded_categorical_cols)}")
    print(f"  Binary features: {len([col for col in binary_cols if col in df.columns])}")
    print(f"  Numerical features for normalization: {len(numerical_cols)}")
    
    # Apply min-max normalization to numerical features
    if fit_scaler:
        print("Fitting new scaler on training data...")
        scaler = MinMaxScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    else:
        print("Using existing scaler for test data...")
        if scaler is None:
            raise ValueError("Scaler must be provided when fit_scaler=False")
        df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    print("Normalization completed.")
    return df, scaler


def preprocess_data(binary_label=False):  # Changed default to False for multi-class
    """Main preprocessing function"""
    print("=== Starting Data Preprocessing ===")
    
    # Load data
    df = load_data()
    
    # Clean data
    df = clean_data(df)
    
    # Display label distribution before processing
    print(f"\nLabel distribution before processing:")
    print(df['label'].value_counts())
    
    # Drop 'difficulty' column (not used in modeling)
    if 'difficulty' in df.columns:
        df = df.drop(columns=['difficulty'])
        print("Dropped 'difficulty' column")
    
    # Multi-class label encoding as per the article
    if not binary_label:
        print("\nConverting labels to 5-class format as per article...")
        
        # Attack mapping from article - maps specific attacks to main categories
        attack_mapping = {
            # DoS attacks
            'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 'smurf': 'DoS', 
            'teardrop': 'DoS', 'apache2': 'DoS', 'udpstorm': 'DoS', 'processtable': 'DoS', 
            'worm': 'DoS',
            # Probe attacks
            'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 
            'mscan': 'Probe', 'saint': 'Probe',
            # R2L attacks
            'guess_passwd': 'R2L', 'ftp_write': 'R2L', 'imap': 'R2L', 'phf': 'R2L', 
            'multihop': 'R2L', 'warezmaster': 'R2L', 'warezclient': 'R2L', 'spy': 'R2L', 
            'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguess': 'R2L', 'snmpgetattack': 'R2L', 
            'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
            # U2R attacks
            'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R', 'perl': 'U2R', 
            'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R'
        }
        
        # Map specific attacks to main categories, keep 'normal' as is
        df['label'] = df['label'].apply(lambda x: attack_mapping.get(x, 'normal'))
        
        # Final numerical mapping for 5 classes
        final_label_mapping = {
            'normal': 0,
            'DoS': 1,
            'Probe': 2,
            'R2L': 3,
            'U2R': 4
        }
        
        df['label'] = df['label'].map(final_label_mapping)
        
        print("Label distribution after 5-class conversion:")
        print(df['label'].value_counts())
        
    else:
        # Binary classification (your original approach)
        print("\nConverting labels to binary format...")
        original_labels = df['label'].unique()
        print(f"Original labels: {original_labels}")
        
        df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
        
        print("Label distribution after binary conversion:")
        print(df['label'].value_counts())
    
    # Use one-hot encoding for categorical features as per article
    categorical_cols = ['protocol_type', 'service', 'flag']
    
    # Verify categorical columns exist
    existing_categorical_cols = [col for col in categorical_cols if col in df.columns]
    if len(existing_categorical_cols) != len(categorical_cols):
        missing_cols = set(categorical_cols) - set(existing_categorical_cols)
        print(f"Warning: Missing categorical columns: {missing_cols}")
    
    print(f"\nApplying one-hot encoding to categorical features: {existing_categorical_cols}")
    
    # Separate label before encoding
    labels = df['label']
    features_df = df.drop('label', axis=1)
    
    # Apply one-hot encoding
    features_encoded = pd.get_dummies(features_df, columns=existing_categorical_cols, dummy_na=False)
    
    # Add label back
    df_encoded = features_encoded.copy()
    df_encoded['label'] = labels
    
    print(f"Features after one-hot encoding: {len(df_encoded.columns)-1}")  # -1 for label
    
    # Identify numerical columns for normalization (exclude binary and label columns)
    binary_cols = ['land', 'logged_in', 'is_host_login', 'is_guest_login', 'label']
    # Get one-hot encoded categorical columns
    encoded_categorical_cols = [col for col in df_encoded.columns if any(cat in col for cat in existing_categorical_cols)]
    
    exclude_cols = binary_cols + encoded_categorical_cols
    numerical_cols = [col for col in df_encoded.columns if col not in exclude_cols]
    
    print(f"\nFeature categorization:")
    print(f"  One-hot encoded categorical features: {len(encoded_categorical_cols)}")
    print(f"  Binary features: {len([col for col in binary_cols if col in df_encoded.columns])}")
    print(f"  Numerical features for normalization: {len(numerical_cols)}")
    
    # Apply min-max normalization to numerical features
    df_normalized, scaler = normalize_numerical_features(df_encoded, numerical_cols)
    
    # Split data with stratification
    print(f"\nSplitting data (80% train, 20% test)...")
    train_df, test_df = train_test_split(df_normalized, test_size=0.2, random_state=42, stratify=df_normalized['label'])
    
    # Save processed data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processed_path = os.path.join(current_dir, '..', 'data', 'processed')
    os.makedirs(processed_path, exist_ok=True)
    
    # Save with appropriate suffix
    suffix = "5class" if not binary_label else "binary"
    train_file = os.path.join(processed_path, f'train_processed_{suffix}.csv')
    test_file = os.path.join(processed_path, f'test_processed_{suffix}.csv')
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"\n=== Preprocessing Complete ===")
    print(f"Training set size: {len(train_df)} ({len(train_df)/len(df_normalized)*100:.1f}%)")
    print(f"Test set size: {len(test_df)} ({len(test_df)/len(df_normalized)*100:.1f}%)")
    print(f"Total features: {len(df_normalized.columns)-1}")  # -1 for label column
    print(f"Classification type: {'5-class' if not binary_label else 'Binary'}")
    print(f"Files saved to: {processed_path}")
    
    return train_df, test_df, scaler


def preprocess_data_article_method(binary_label=False):
    """
    Main preprocessing function following the exact article methodology:
    - Load train and test sets separately
    - Process them independently 
    - Fit scaler on training data only
    - Apply same scaler to test data
    """
    print("=== Starting Data Preprocessing (Article Method) ===")
    
    # Load train and test data separately
    train_df, test_df = load_data_separately()
    
    # Process training data
    print("\n" + "="*50)
    print("PROCESSING TRAINING DATA")
    print("="*50)
    
    train_df = preprocess_dataset(train_df, "Training")
    train_df = apply_label_encoding(train_df, binary_label)
    train_df = apply_feature_encoding(train_df)
    train_df, scaler = apply_normalization(train_df, fit_scaler=True)
    
    # Process test data using the same transformations
    print("\n" + "="*50)
    print("PROCESSING TEST DATA")
    print("="*50)
    
    test_df = preprocess_dataset(test_df, "Test")
    test_df = apply_label_encoding(test_df, binary_label)
    test_df = apply_feature_encoding(test_df)
    
    # Ensure test data has same columns as training data
    # Add missing columns with zeros
    missing_cols = set(train_df.columns) - set(test_df.columns)
    for col in missing_cols:
        if col != 'label':  # Don't add label column
            test_df[col] = 0
            print(f"Added missing column '{col}' to test data with zeros")
    
    # Remove extra columns from test data
    extra_cols = set(test_df.columns) - set(train_df.columns)
    for col in extra_cols:
        if col != 'label':  # Don't remove label column
            test_df = test_df.drop(columns=[col])
            print(f"Removed extra column '{col}' from test data")
    
    # Reorder columns to match training data
    test_df = test_df[train_df.columns]
    
    # Apply normalization to test data using training scaler
    test_df, _ = apply_normalization(test_df, scaler=scaler, fit_scaler=False)
    
    # Save processed data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processed_path = os.path.join(current_dir, '..', 'data', 'processed')
    os.makedirs(processed_path, exist_ok=True)
    
    # Save with appropriate suffix
    suffix = "5class" if not binary_label else "binary"
    train_file = os.path.join(processed_path, f'train_processed_{suffix}_article.csv')
    test_file = os.path.join(processed_path, f'test_processed_{suffix}_article.csv')
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"\n=== Article Method Preprocessing Complete ===")
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    print(f"Total features: {len(train_df.columns)-1}")  # -1 for label column
    print(f"Classification type: {'5-class' if not binary_label else 'Binary'}")
    print(f"Files saved to: {processed_path}")
    
    return train_df, test_df, scaler


def validate_preprocessing():
    """Validation function to check if preprocessing works correctly"""
    try:
        print("=== Validating Preprocessing (5-class) ===")
        train_df, test_df, scaler = preprocess_data(binary_label=False)
        
        # Basic validation checks
        assert len(train_df) > 0, "Training set is empty"
        assert len(test_df) > 0, "Test set is empty"
        assert 'label' in train_df.columns, "Label column missing"
        
        # Check for 5-class labels (0-4)
        unique_labels = sorted(train_df['label'].unique())
        expected_labels = [0, 1, 2, 3, 4]
        assert unique_labels == expected_labels, f"Expected labels {expected_labels}, got {unique_labels}"
        
        # Check for missing values
        assert train_df.isnull().sum().sum() == 0, "Training set has missing values"
        assert test_df.isnull().sum().sum() == 0, "Test set has missing values"
        
        print("✓ All validation checks passed!")
        print("✓ 5-class preprocessing is working correctly!")
        
        # Also test binary classification
        print("\n=== Validating Preprocessing (Binary) ===")
        train_df_bin, test_df_bin, scaler_bin = preprocess_data(binary_label=True)
        
        # Check for binary labels (0-1)
        unique_labels_bin = sorted(train_df_bin['label'].unique())
        expected_labels_bin = [0, 1]
        assert unique_labels_bin == expected_labels_bin, f"Expected binary labels {expected_labels_bin}, got {unique_labels_bin}"
        
        print("✓ Binary preprocessing also working correctly!")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation failed: {str(e)}")
        return False


if __name__ == '__main__':
    # Run validation
    if validate_preprocessing():
        print("\n=== Ready for Feature Selection and Classification ===")
    else:
        print("\n=== Please fix preprocessing issues before proceeding ===")
