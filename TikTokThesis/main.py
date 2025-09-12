from src.preprocessing import preprocess_dataset, save_preprocessed_dataset
import pandas


if __name__ == "__main__":
    dataset_path = "data/testing.csv"  # adjust this
    output_path = "data/testing_results.csv"
    df = preprocess_dataset(dataset_path)

    for i, row in df.iterrows():
        print(f"Transcript {i+1}: {row['transcript']}")
        print(f"Tokens {i+1}: {row['tokens']}\n")
    
    save_preprocessed_dataset(df,output_path)