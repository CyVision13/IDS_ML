import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, pearsonr, ConstantInputWarning
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from pathlib import Path
import warnings
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try importing custom plotting function
try:
    from code.utils.plotting_functions import plot_feature_rankings
except ImportError as e:
    print(f"Import error: {e}")
    plot_feature_rankings = None

# Paths
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"


def load_data():
    """Load the correctly preprocessed, INTEGER-ENCODED training data for feature ranking"""
    # This is the correct filename from your directory
    correct_file_path = PROCESSED_DATA_DIR / "train_processed_5class_integer.csv"

    print(f"📂 Loading preprocessed data from: {correct_file_path}")
    train_df = pd.read_csv(correct_file_path)

    print(f"Loaded training data for feature selection: {train_df.shape}")
    return train_df


def normalize_features(X):
    """Apply Min-Max normalization to features (as done in the article)"""
    print("🔧 Applying Min-Max normalization to features...")
    
    scaler = MinMaxScaler()
    X_normalized = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    print(f"   - Features normalized to range [0, 1]")
    print(f"   - Sample ranges after normalization:")
    for col in X_normalized.columns[:3]:  # Show first 3 features as example
        print(f"     {col}: [{X_normalized[col].min():.3f}, {X_normalized[col].max():.3f}]")
    
    return X_normalized, scaler


def remove_constant_features(df):
    """Remove features with only one unique value"""
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        print(f"Removing constant features: {constant_cols}")
        return df.drop(columns=constant_cols)
    else:
        print("No constant features found")
        return df


def chi_square_ranking(X, y):
    """Calculate Chi-square statistics for feature ranking using sklearn's chi2"""
    print("🔹 Calculating Chi-square rankings...")
    
    # Ensure all values are non-negative for chi2 test
    X_positive = X.copy()
    
    # Shift negative values to make them positive (though shouldn't be needed after normalization)
    for col in X_positive.columns:
        min_val = X_positive[col].min()
        if min_val < 0:
            X_positive[col] = X_positive[col] - min_val
    
    try:
        # Use sklearn's chi2 function which is more robust
        chi2_scores, p_values = chi2(X_positive, y)
        
        ranking = pd.DataFrame({
            'Feature': X.columns,
            'Chi2_statistic': chi2_scores,
            'P_value': p_values
        }).sort_values('Chi2_statistic', ascending=False).reset_index(drop=True)
        
        print(f"✓ Chi-square ranking completed for {len(ranking)} features")
        return ranking
        
    except Exception as e:
        print(f"Error in chi-square calculation: {e}")
        # Fallback to manual calculation
        return chi_square_ranking_manual(X, y)


def chi_square_ranking_manual(X, y):
    """Manual chi-square calculation as fallback"""
    print("Using manual chi-square calculation...")
    chi2_stats = []
    p_values = []
    valid_features = []

    for feature in X.columns:
        try:
            # Discretize continuous features for chi-square test
            if X[feature].nunique() > 10:  # If too many unique values
                # Bin into 5 quantiles
                X_binned = pd.qcut(X[feature], q=5, duplicates='drop')
            else:
                X_binned = X[feature]
            
            # Create contingency table
            contingency_table = pd.crosstab(X_binned, y)
            
            # Skip if contingency table is too small
            if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                continue
                
            chi2_stat, p, _, _ = chi2_contingency(contingency_table)
            chi2_stats.append(chi2_stat)
            p_values.append(p)
            valid_features.append(feature)
            
        except Exception as e:
            print(f"Skipping {feature}: {e}")
            continue

    ranking = pd.DataFrame({
        'Feature': valid_features,
        'Chi2_statistic': chi2_stats,
        'P_value': p_values
    }).sort_values('Chi2_statistic', ascending=False).reset_index(drop=True)

    print(f"✓ Manual chi-square ranking completed for {len(ranking)} features")
    return ranking


def mad_ranking(X):
    """Calculate Mean Absolute Deviation for feature ranking"""
    print("🔹 Calculating MAD rankings...")
    
    # Select only numeric columns
    X_numeric = X.select_dtypes(include=[np.number])
    print(f"Processing {len(X_numeric.columns)} numeric features for MAD")

    # Calculate MAD for each feature
    mad_values = []
    feature_names = []
    
    for col in X_numeric.columns:
        try:
            # Calculate mean
            mean_val = X_numeric[col].mean()
            # Calculate mean absolute deviation
            mad_val = np.mean(np.abs(X_numeric[col] - mean_val))
            
            mad_values.append(mad_val)
            feature_names.append(col)
            
        except Exception as e:
            print(f"Error calculating MAD for {col}: {e}")
            continue

    ranking = pd.DataFrame({
        'Feature': feature_names,
        'MAD': mad_values
    }).sort_values('MAD', ascending=False).reset_index(drop=True)

    print(f"✓ MAD ranking completed for {len(ranking)} features")
    return ranking


def pcc_ranking(X, y):
    """Calculate Pearson Correlation Coefficient for feature ranking"""
    print("🔹 Calculating PCC rankings...")
    
    # Ensure y is numeric
    if y.dtype == 'object':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        print("Encoded target labels to numeric values")
    else:
        y_encoded = y

    pcc_values = []
    p_values = []
    valid_features = []

    for feature in X.columns:
        try:
            # Skip constant features
            if X[feature].nunique() <= 1:
                print(f"Skipping constant feature: {feature}")
                continue
            
            # Skip non-numeric features
            if not pd.api.types.is_numeric_dtype(X[feature]):
                print(f"Skipping non-numeric feature: {feature}")
                continue
                
            # Calculate correlation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConstantInputWarning)
                corr, p = pearsonr(X[feature], y_encoded)
                
            # Handle NaN correlations
            if np.isnan(corr):
                print(f"Skipping {feature}: NaN correlation")
                continue
                
            pcc_values.append(abs(corr))  # Use absolute value for ranking
            p_values.append(p)
            valid_features.append(feature)
            
        except Exception as e:
            print(f"Error processing {feature}: {e}")
            continue

    ranking = pd.DataFrame({
        'Feature': valid_features,
        'PCC': pcc_values,
        'P_value': p_values
    }).sort_values('PCC', ascending=False).reset_index(drop=True)

    print(f"✓ PCC ranking completed for {len(ranking)} features")
    return ranking


def save_rankings(chi2_rank, mad_rank, pcc_rank):
    """Save ranking results to CSV files and create visualizations"""
    # Create directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Define file paths
    chi2_path = RESULTS_DIR / "chi2_ranking.csv"
    mad_path = RESULTS_DIR / "mad_ranking.csv"
    pcc_path = RESULTS_DIR / "pcc_ranking.csv"

    # Save CSV files
    chi2_rank.to_csv(chi2_path, index=False)
    mad_rank.to_csv(mad_path, index=False)
    pcc_rank.to_csv(pcc_path, index=False)

    print("\n📊 Saved ranking tables:")
    print(f"- Chi-square: {chi2_path} ({len(chi2_rank)} features)")
    print(f"- MAD: {mad_path} ({len(mad_rank)} features)")
    print(f"- PCC: {pcc_path} ({len(pcc_rank)} features)")

    # Create visualization if plotting function is available
    if plot_feature_rankings is not None:
        try:
            ranking_files = {
                "Chi-square Ranking": chi2_path,
                "MAD Ranking": mad_path,
                "PCC Ranking": pcc_path
            }

            fig = plot_feature_rankings(ranking_files)
            plot_path = FIGURE_DIR / "feature_rankings.png"
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"- Visualization: {plot_path}")
        except Exception as e:
            print(f"Warning: Could not create visualization: {e}")
    else:
        print("- Visualization: Skipped (plotting function not available)")


def display_top_features(chi2_rank, mad_rank, pcc_rank, top_n=10):
    """Display top N features from each ranking method"""
    print(f"\n=== Top {top_n} Features by Each Method ===")
    
    print(f"\n🏆 Chi-square (Top {top_n}):")
    for i, row in chi2_rank.head(top_n).iterrows():
        print(f"{i+1:2d}. {row['Feature']:30s} (χ²={row['Chi2_statistic']:.4f})")
    
    print(f"\n🏆 MAD (Top {top_n}):")
    for i, row in mad_rank.head(top_n).iterrows():
        print(f"{i+1:2d}. {row['Feature']:30s} (MAD={row['MAD']:.4f})")
    
    print(f"\n🏆 PCC (Top {top_n}):")
    for i, row in pcc_rank.head(top_n).iterrows():
        print(f"{i+1:2d}. {row['Feature']:30s} (|r|={row['PCC']:.4f})")


def feature_selection():
    """Main feature selection function - Phase 2 of the pipeline"""
    print("=" * 60)
    print("🎯 PHASE 2: FEATURE RANKING (Article Method)")
    print("=" * 60)
    
    # Load ONLY training data for feature selection
    print("📂 Loading preprocessed training data...")
    train_df = load_data()

    # Separate features and target
    if 'label' in train_df.columns:
        X = train_df.drop(columns=['label'])
        y = train_df['label']
    else:
        # Assume last column is the target
        X = train_df.iloc[:, :-1]
        y = train_df.iloc[:, -1]

    print(f"📊 Dataset info:")
    print(f"   - Features: {X.shape[1]}")
    print(f"   - Training samples: {X.shape[0]}")
    print(f"   - Target distribution: {y.value_counts().to_dict()}")

    # Remove constant features
    print("\n🧹 Removing constant features...")
    X_clean = remove_constant_features(X)
    print(f"   - Features after cleaning: {X_clean.shape[1]}")

    # Apply Min-Max normalization as done in the article
    print("\n🔄 Normalizing features to [0, 1] range...")
    X_normalized, _ = normalize_features(X_clean)
    print(f"   - Normalization completed")

    # Calculate rankings
    chi2_rank = chi_square_ranking(X_normalized, y)
    mad_rank = mad_ranking(X_normalized)
    pcc_rank = pcc_ranking(X_normalized, y)

    # Save results
    save_rankings(chi2_rank, mad_rank, pcc_rank)
    
    # Display summary
    display_top_features(chi2_rank, mad_rank, pcc_rank)
    
    print("\n✅ Feature ranking completed successfully!")
    print("Ready for Phase 2b: Fuzzy TOPSIS consensus ranking")
    
    return chi2_rank, mad_rank, pcc_rank


if __name__ == "__main__":
    chi2_rank, mad_rank, pcc_rank = feature_selection()
