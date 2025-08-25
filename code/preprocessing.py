
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


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
    print(f"Initial dataset shape: {df.shape}")
    
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


def apply_integer_encoding(df, categorical_cols=None, encoders=None, fit_encoders=True):
    """Apply integer encoding to categorical features (for feature ranking)"""
    if categorical_cols is None:
        categorical_cols = ['protocol_type', 'service', 'flag']
    
    # Verify categorical columns exist
    existing_categorical_cols = [col for col in categorical_cols if col in df.columns]
    
    print(f"Applying integer encoding to categorical features: {existing_categorical_cols}")
    
    if fit_encoders:
        encoders = {}
        for col in existing_categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    else:
        if encoders is None:
            raise ValueError("Encoders must be provided when fit_encoders=False")
        for col in existing_categorical_cols:
            if col in encoders:
                df[col] = encoders[col].transform(df[col])
    
    print(f"Integer encoding completed. Features: {len(df.columns)-1}")  # -1 for label
    
    return df, encoders


def apply_onehot_encoding(df, categorical_cols=None):
    """Apply one-hot encoding to categorical features (for ML algorithms)"""
    if categorical_cols is None:
        categorical_cols = ['protocol_type', 'service', 'flag']
    
    # Verify categorical columns exist
    existing_categorical_cols = [col for col in categorical_cols if col in df.columns]
    
    print(f"Applying one-hot encoding to categorical features: {existing_categorical_cols}")
    
    # Separate label before encoding
    labels = df['label']
    features_df = df.drop('label', axis=1)
    
    # Apply one-hot encoding
    features_encoded = pd.get_dummies(features_df, columns=existing_categorical_cols, dummy_na=False)
    
    # Add label back
    df_encoded = features_encoded.copy()
    df_encoded['label'] = labels
    
    print(f"One-hot encoding completed. Features: {len(df_encoded.columns)-1}")  # -1 for label
    
    return df_encoded


def apply_normalization(df, scaler=None, fit_scaler=True, encoding_type='integer'):
    """Apply min-max normalization to numerical features"""
    # Identify numerical columns for normalization
    binary_cols = ['land', 'logged_in', 'is_host_login', 'is_guest_login', 'label']
    
    if encoding_type == 'onehot':
        # For one-hot encoded data, exclude one-hot encoded categorical columns
        categorical_prefixes = ['protocol_type', 'service', 'flag']
        encoded_categorical_cols = [col for col in df.columns if any(cat in col for cat in categorical_prefixes)]
        exclude_cols = binary_cols + encoded_categorical_cols
    else:
        # For integer encoded data, include categorical columns in normalization
        exclude_cols = binary_cols
    
    numerical_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Normalizing {len(numerical_cols)} numerical features...")
    
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


def preprocess_both_encodings(binary_label=False):
    """
    Main preprocessing function that creates both integer-encoded and one-hot encoded datasets
    
    Returns:
    - Integer-encoded datasets (for feature ranking): 41 features
    - One-hot encoded datasets (for ML algorithms): 122+ features
    """
    print("=== Starting Dual Preprocessing (Integer + One-Hot Encoding) ===")
    
    # Load train and test data separately
    train_df_raw, test_df_raw = load_data_separately()
    
    # ========================================
    # PROCESS INTEGER-ENCODED DATASETS (41 features)
    # ========================================
    print("\n" + "="*60)
    print("CREATING INTEGER-ENCODED DATASETS (FOR FEATURE RANKING)")
    print("="*60)
    
    # Process training data with integer encoding
    train_df_int = train_df_raw.copy()
    train_df_int = clean_data(train_df_int)
    
    # Drop difficulty column
    if 'difficulty' in train_df_int.columns:
        train_df_int = train_df_int.drop(columns=['difficulty'])
    
    train_df_int = apply_label_encoding(train_df_int, binary_label)
    train_df_int, encoders = apply_integer_encoding(train_df_int, fit_encoders=True)
    train_df_int, scaler_int = apply_normalization(train_df_int, fit_scaler=True, encoding_type='integer')
    
    # Process test data with integer encoding
    test_df_int = test_df_raw.copy()
    test_df_int = clean_data(test_df_int)
    
    # Drop difficulty column
    if 'difficulty' in test_df_int.columns:
        test_df_int = test_df_int.drop(columns=['difficulty'])
    
    test_df_int = apply_label_encoding(test_df_int, binary_label)
    test_df_int, _ = apply_integer_encoding(test_df_int, encoders=encoders, fit_encoders=False)
    test_df_int, _ = apply_normalization(test_df_int, scaler=scaler_int, fit_scaler=False, encoding_type='integer')
    
    # ========================================
    # PROCESS ONE-HOT ENCODED DATASETS (122+ features)
    # ========================================
    print("\n" + "="*60)
    print("CREATING ONE-HOT ENCODED DATASETS (FOR ML ALGORITHMS)")
    print("="*60)
    
    # Process training data with one-hot encoding
    train_df_oh = train_df_raw.copy()
    train_df_oh = clean_data(train_df_oh)
    
    # Drop difficulty column
    if 'difficulty' in train_df_oh.columns:
        train_df_oh = train_df_oh.drop(columns=['difficulty'])
    
    train_df_oh = apply_label_encoding(train_df_oh, binary_label)
    train_df_oh = apply_onehot_encoding(train_df_oh)
    train_df_oh, scaler_oh = apply_normalization(train_df_oh, fit_scaler=True, encoding_type='onehot')
    
    # Process test data with one-hot encoding
    test_df_oh = test_df_raw.copy()
    test_df_oh = clean_data(test_df_oh)
    
    # Drop difficulty column
    if 'difficulty' in test_df_oh.columns:
        test_df_oh = test_df_oh.drop(columns=['difficulty'])
    
    test_df_oh = apply_label_encoding(test_df_oh, binary_label)
    test_df_oh = apply_onehot_encoding(test_df_oh)
    
    # Ensure test data has same columns as training data
    # Add missing columns with zeros
    missing_cols = set(train_df_oh.columns) - set(test_df_oh.columns)
    for col in missing_cols:
        if col != 'label':  # Don't add label column
            test_df_oh[col] = 0
            print(f"Added missing column '{col}' to test data with zeros")
    
    # Remove extra columns from test data
    extra_cols = set(test_df_oh.columns) - set(train_df_oh.columns)
    for col in extra_cols:
        if col != 'label':  # Don't remove label column
            test_df_oh = test_df_oh.drop(columns=[col])
            print(f"Removed extra column '{col}' from test data")
    
    # Reorder columns to match training data
    test_df_oh = test_df_oh[train_df_oh.columns]
    
    # Apply normalization to test data using training scaler
    test_df_oh, _ = apply_normalization(test_df_oh, scaler=scaler_oh, fit_scaler=False, encoding_type='onehot')
    
    # Save processed data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processed_path = os.path.join(current_dir, '..', 'data', 'processed')
    os.makedirs(processed_path, exist_ok=True)
    
    # Save integer-encoded datasets
    int_suffix = "5class" if not binary_label else "binary"
    train_int_file = os.path.join(processed_path, f'train_processed_{int_suffix}_integer.csv')
    test_int_file = os.path.join(processed_path, f'test_processed_{int_suffix}_integer.csv')
    
    train_df_int.to_csv(train_int_file, index=False)
    test_df_int.to_csv(test_int_file, index=False)
    
    # Save one-hot encoded datasets
    oh_suffix = "5class" if not binary_label else "binary"
    train_oh_file = os.path.join(processed_path, f'train_processed_{oh_suffix}_onehot.csv')
    test_oh_file = os.path.join(processed_path, f'test_processed_{oh_suffix}_onehot.csv')
    
    train_df_oh.to_csv(train_oh_file, index=False)
    test_df_oh.to_csv(test_oh_file, index=False)
    
    print(f"\n=== Dual Preprocessing Complete ===")
    print(f"Integer-encoded datasets:")
    print(f"  Training set size: {len(train_df_int)}")
    print(f"  Test set size: {len(test_df_int)}")
    print(f"  Features: {len(train_df_int.columns)-1}")
    print(f"One-hot encoded datasets:")
    print(f"  Training set size: {len(train_df_oh)}")
    print(f"  Test set size: {len(test_df_oh)}")
    print(f"  Features: {len(train_df_oh.columns)-1}")
    print(f"Classification type: {'5-class' if not binary_label else 'Binary'}")
    print(f"Files saved to: {processed_path}")
    
    return train_df_int, test_df_int, train_df_oh, test_df_oh, scaler_int, scaler_oh


def validate_dual_preprocessing():
    """Validation function to check if dual preprocessing works correctly"""
    try:
        print("=== Validating Dual Preprocessing (5-class) ===")
        train_int, test_int, train_oh, test_oh, scaler_int, scaler_oh = preprocess_both_encodings(binary_label=False)
        
        # Basic validation checks for integer-encoded data
        assert len(train_int) > 0, "Integer training set is empty"
        assert len(test_int) > 0, "Integer test set is empty"
        assert 'label' in train_int.columns, "Integer label column missing"
        
        # Check for 5-class labels (0-4)
        unique_labels_int = sorted(train_int['label'].unique())
        expected_labels_int = [0, 1, 2, 3, 4]
        assert unique_labels_int == expected_labels_int, f"Expected integer labels {expected_labels_int}, got {unique_labels_int}"
        
        # Check for missing values
        assert train_int.isnull().sum().sum() == 0, "Integer training set has missing values"
        assert test_int.isnull().sum().sum() == 0, "Integer test set has missing values"
        
        # Basic validation checks for one-hot encoded data
        assert len(train_oh) > 0, "One-hot training set is empty"
        assert len(test_oh) > 0, "One-hot test set is empty"
        assert 'label' in train_oh.columns, "One-hot label column missing"
        
        # Check for 5-class labels (0-4)
        unique_labels_oh = sorted(train_oh['label'].unique())
        assert unique_labels_oh == expected_labels_int, f"Expected one-hot labels {expected_labels_int}, got {unique_labels_oh}"
        
        # Check for missing values
        assert train_oh.isnull().sum().sum() == 0, "One-hot training set has missing values"
        assert test_oh.isnull().sum().sum() == 0, "One-hot test set has missing values"
        
        print("✓ All validation checks passed!")
        print("✓ Dual preprocessing (5-class) is working correctly!")
        
        # Also test binary classification
        print("\n=== Validating Dual Preprocessing (Binary) ===")
        train_int_bin, test_int_bin, train_oh_bin, test_oh_bin, scaler_int_bin, scaler_oh_bin = preprocess_both_encodings(binary_label=True)
        
        # Check for binary labels (0-1)
        unique_labels_int_bin = sorted(train_int_bin['label'].unique())
        expected_labels_bin = [0, 1]
        assert unique_labels_int_bin == expected_labels_bin, f"Expected binary labels {expected_labels_bin}, got {unique_labels_int_bin}"
        
        print("✓ Binary dual preprocessing also working correctly!")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation failed: {str(e)}")
        return False


if __name__ == '__main__':
    # Run validation
    if validate_dual_preprocessing():
        print("\n=== Ready for Feature Selection and Classification ===")
    else:
        print("\n=== Please fix preprocessing issues before proceeding ===")
