import argparse
import pandas as pd
import random

def load_dataset(filepath):
    # Load the CSV file
    df = pd.read_csv(filepath, sep="\t", engine="python")

    # Clean column names (in case of extra spaces)
    df.columns = [col.strip() for col in df.columns]

    return df

def preprocess_data(df):
    # Keep only numerical features of interest
    features = [
        'Ratings',
        'Budget (in Million USD)',
        'Number of Episodes',
        'Duration per Episode (minutes)'
    ]
    df = df[features].copy()
    
    # Convert all to numeric, drop missing values
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)
    return df

def split_three_way(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    total = len(df_shuffled)
    
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)
    
    train = df_shuffled[:train_end]
    val = df_shuffled[train_end:val_end]
    test = df_shuffled[val_end:]
    
    return train, val, test

def analyze_statistics(df):
    print("\n=== Dataset Statistics ===")
    print("\nMean:\n", df.mean(numeric_only=True))
    print("\nMax:\n", df.max(numeric_only=True))
    print("\nMin:\n", df.min(numeric_only=True))
    print("\nCorrelation Matrix:\n", df.corr(numeric_only=True))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to dataset file")
    args = parser.parse_args()

    df = load_dataset(args.file)
    df_clean = preprocess_data(df)

    analyze_statistics(df_clean)

    train, val, test = split_three_way(df_clean)

    print(f"\n=== Split Sizes ===")
    print(f"Train size: {len(train)}")
    print(f"Validation size: {len(val)}")
    print(f"Test size: {len(test)}")

if __name__ == "__main__":
    main()
