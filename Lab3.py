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
    # Keep only relevant columns for prediction
    features = ['Ratings', 'Budget (in Million USD)', 'Number of Episodes', 'Duration per Episode (minutes)']
    df = df[features].copy()

    # Convert to appropriate types
    df = df.apply(pd.to_numeric, errors='coerce')

    # Drop rows with missing values
    df = df.dropna()

    return df

def splitData(df, train_ratio=0.8):
    """
    Sequential split: First 80% for training, rest for testing
    """
    split_index = int(len(df) * train_ratio)
    train_data = df.iloc[:split_index]
    test_data = df.iloc[split_index:]
    return train_data, test_data

def splitDataRandom(df, train_ratio=0.8):
    """
    Random split: 80% training, 20% testing
    """
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_index = int(len(df_shuffled) * train_ratio)
    train_data = df_shuffled.iloc[:split_index]
    test_data = df_shuffled.iloc[split_index:]
    return train_data, test_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to dataset file")
    args = parser.parse_args()

    df = load_dataset(args.file)
    df_clean = preprocess_data(df)

    print("=== Sequential Split ===")
    train_seq, test_seq = splitData(df_clean)
    print(f"Train size: {len(train_seq)} | Test size: {len(test_seq)}")

    print("\n=== Random Split ===")
    train_rand, test_rand = splitDataRandom(df_clean)
    print(f"Train size: {len(train_rand)} | Test size: {len(test_rand)}")

if __name__ == "__main__":
    main()
